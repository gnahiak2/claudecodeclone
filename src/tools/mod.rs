mod diff;
mod patch;
mod repo;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::providers::ToolDefinition;
use crate::safe_fs::{DirHandle, ReadResult};
use crate::util::{safe_join, truncate_bytes};
pub use diff::{ChangeStatus, ChangedFile, DiffReport, compute_diff};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolRole {
    Planner,
    Executor,
}

#[derive(Debug)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

#[derive(Debug)]
pub struct ToolContext {
    pub root: PathBuf,
    pub baseline: Option<PathBuf>,
    pub max_read_bytes: usize,
    pub max_output_bytes: usize,
    /// Paths the approved plan permits the executor to write. Reads are not
    /// restricted by this; mutations are.
    scope: Option<Vec<String>>,
    handle: DirHandle,
}

impl ToolContext {
    pub fn new(
        root: impl Into<PathBuf>,
        baseline: Option<PathBuf>,
        max_read_bytes: usize,
        max_output_bytes: usize,
    ) -> Result<Self> {
        let root = root.into();
        let root = root
            .canonicalize()
            .with_context(|| format!("failed to resolve workspace {}", root.display()))?;
        let baseline = match baseline {
            Some(path) => Some(
                path.canonicalize()
                    .with_context(|| format!("failed to resolve baseline {}", path.display()))?,
            ),
            None => None,
        };
        let handle = DirHandle::open_root(&root)?;
        Ok(Self {
            root,
            baseline,
            max_read_bytes,
            max_output_bytes,
            scope: None,
            handle,
        })
    }

    /// Restrict mutations to the files approved in the plan. Entries ending in
    /// `/` are treated as directories that may receive new files.
    pub fn with_scope(mut self, scope: Vec<String>) -> Self {
        self.scope = Some(scope);
        self
    }

    pub fn path_is_writable(&self, relative: &str) -> bool {
        let normalized = normalize_relative(relative);
        let Some(scope) = &self.scope else {
            return true;
        };
        scope.iter().any(|entry| {
            let is_directory = entry.ends_with('/');
            let entry = normalize_relative(entry.trim_end_matches('/'));
            if is_directory {
                normalized == entry || normalized.starts_with(&format!("{entry}/"))
            } else {
                normalized == entry
            }
        })
    }

    pub fn read_bounded(&self, relative: &str, limit: usize) -> Result<ReadResult> {
        self.handle.read(relative, limit)
    }

    pub fn write_file(&self, relative: &str, data: &[u8], mode: Option<u32>) -> Result<()> {
        if !self.path_is_writable(relative) {
            bail!("path {relative} is outside the approved plan file scope");
        }
        self.handle.write(relative, data, mode)
    }

    pub fn remove_file(&self, relative: &str) -> Result<()> {
        if !self.path_is_writable(relative) {
            bail!("path {relative} is outside the approved plan file scope");
        }
        self.handle.remove(relative)
    }

    pub fn mode_of(&self, relative: &str) -> Option<u32> {
        self.handle.mode_of(relative).ok()
    }

    pub fn exists(&self, relative: &str) -> bool {
        self.handle.metadata(relative).is_ok()
    }

    pub fn definitions(role: ToolRole) -> Vec<ToolDefinition> {
        let mut tools = vec![
            ToolDefinition {
                name: "list_files".to_owned(),
                description: "List files and directories below a relative path. Honors ignore files and excludes secrets.".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative directory path, defaults to ."},
                        "max_depth": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Maximum depth, defaults to 2."}
                    },
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "read_file".to_owned(),
                description: "Read a UTF-8 text file inside the workspace. Files may be truncated at the configured limit.".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative file path."},
                        "start_line": {"type": "integer", "minimum": 1, "description": "Optional 1-based first line."},
                        "end_line": {"type": "integer", "minimum": 1, "description": "Optional inclusive last line."}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }),
            },
            ToolDefinition {
                name: "search".to_owned(),
                description: "Search text files for a literal or regular expression. Returns bounded matching lines.".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text or regular expression to find."},
                        "regex": {"type": "boolean", "description": "Treat query as a regular expression. Defaults to false."},
                        "path": {"type": "string", "description": "Optional relative directory or file to search."},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Maximum matches, defaults to 40."}
                    },
                    "required": ["query"],
                    "additionalProperties": false
                }),
            },
        ];
        if role == ToolRole::Executor {
            tools.push(ToolDefinition {
                name: "apply_patch".to_owned(),
                description: "Apply atomic text edits in the staged workspace. Existing files require expected_hash; set old_text to replace one exact occurrence, or omit old_text to replace the whole file. Set create=true for a new file or delete=true to remove one.".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "edits": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 20,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "expected_hash": {"type": ["string", "null"], "description": "BLAKE3 hex hash from read_file/apply_patch. Required for edits and deletes."},
                                    "old_text": {"type": ["string", "null"]},
                                    "new_text": {"type": "string"},
                                    "create": {"type": "boolean"},
                                    "delete": {"type": "boolean"}
                                },
                                "required": ["path"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["edits"],
                    "additionalProperties": false
                }),
            });
            tools.push(ToolDefinition {
                name: "diff".to_owned(),
                description: "Show the unified diff between the original snapshot and the current staged workspace.".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
            });
        }
        tools
    }

    pub fn execute(&self, role: ToolRole, name: &str, arguments: &Value) -> ToolResult {
        let result: Result<String> = match name {
            "list_files" => repo::list_files(self, arguments),
            "read_file" => repo::read_file(self, arguments),
            "search" => repo::search(self, arguments),
            "apply_patch" if role == ToolRole::Executor => patch::apply_patch(self, arguments),
            "diff" if role == ToolRole::Executor => diff::tool_diff(self, arguments),
            "apply_patch" | "diff" => {
                Err(anyhow::anyhow!("tool {name} is not available to this role"))
            }
            _ => Err(anyhow::anyhow!("unknown tool {name}")),
        };
        match result {
            Ok(content) => ToolResult::ok(truncate_bytes(&content, self.max_output_bytes)),
            Err(error) => ToolResult::error(truncate_bytes(
                &format!("tool error: {error:#}"),
                self.max_output_bytes,
            )),
        }
    }
}

pub(crate) fn validate_ancestors(root: &Path, path: &Path) -> Result<()> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("{} is outside the workspace", path.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    bail!("refusing to traverse symlink {}", current.display());
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

pub(crate) fn validate_search_root(ctx: &ToolContext, relative: &str) -> Result<PathBuf> {
    if relative == "." {
        return Ok(ctx.root.clone());
    }
    let path = safe_join(&ctx.root, relative)?;
    validate_ancestors(&ctx.root, &path)?;
    let metadata =
        std::fs::symlink_metadata(&path).with_context(|| format!("path not found: {relative}"))?;
    if metadata.file_type().is_symlink() {
        bail!("refusing to follow a symlink: {relative}");
    }
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(&ctx.root) {
        bail!("path escapes the workspace");
    }
    Ok(canonical)
}

pub fn normalize_relative(relative: &str) -> String {
    let mut parts = Vec::new();
    for component in Path::new(relative).components() {
        if let std::path::Component::Normal(name) = component {
            parts.push(name.to_string_lossy().to_string());
        }
    }
    let joined = parts.join("/");
    #[cfg(target_os = "macos")]
    {
        joined.to_lowercase()
    }
    #[cfg(not(target_os = "macos"))]
    {
        joined
    }
}

pub(crate) fn require_string(arguments: &Value, name: &str) -> Result<String> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .with_context(|| format!("missing string argument {name}"))
}

pub(crate) fn optional_u64(arguments: &Value, name: &str) -> Result<Option<u64>> {
    match arguments.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .with_context(|| format!("{name} must be a positive integer")),
    }
}

pub(crate) fn optional_bool(arguments: &Value, name: &str) -> Result<bool> {
    match arguments.get(name) {
        None | Some(Value::Null) => Ok(false),
        Some(value) => value
            .as_bool()
            .with_context(|| format!("{name} must be a boolean")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> (tempfile::TempDir, ToolContext) {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "hello\n").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        (directory, context)
    }

    #[test]
    fn planner_cannot_apply_patches() {
        let (_directory, context) = context();
        let result = context.execute(
            ToolRole::Planner,
            "apply_patch",
            &json!({"edits": [{"path": "a.txt", "create": true, "new_text": "x"}]}),
        );
        assert!(result.is_error);
        assert!(result.content.contains("not available"));
    }

    #[test]
    fn directory_scope_allows_children_only_beneath_it() {
        let (_directory, context) = context();
        let context = context.with_scope(vec!["src/".to_owned()]);
        assert!(context.path_is_writable("src/new.rs"));
        assert!(context.path_is_writable("src/nested/new.rs"));
        assert!(!context.path_is_writable("srcs/file.rs"));
        assert!(!context.path_is_writable("other/file.rs"));
    }

    #[test]
    fn rejects_secret_and_traversing_reads() {
        let (_directory, context) = context();
        assert!(
            context
                .execute(ToolRole::Planner, "read_file", &json!({"path": ".env"}))
                .is_error
        );
        assert!(
            context
                .execute(ToolRole::Planner, "read_file", &json!({"path": "../x"}))
                .is_error
        );
    }
}
