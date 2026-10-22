use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::{ToolContext, optional_u64, require_string, validate_search_root};
use crate::repo_fs::{relative_string, walker};
use crate::util::truncate_bytes;

const MAX_LISTING_ENTRIES: usize = 2000;
const MAX_SEARCH_FILES: usize = 20_000;

pub(super) fn list_files(ctx: &ToolContext, arguments: &Value) -> Result<String> {
    let relative = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
    let base = validate_search_root(ctx, relative)?;
    if !base.is_dir() {
        bail!("not a directory: {relative}");
    }
    let depth = optional_u64(arguments, "max_depth")?
        .unwrap_or(2)
        .clamp(1, 8) as usize;
    let mut lines = Vec::new();
    let mut truncated = false;
    // Root at the repository so ignore rules apply consistently to every
    // listing, even when the requested path is a subdirectory.
    for entry in walker(&ctx.root)? {
        let entry = entry?;
        if entry.depth() == 0 {
            continue;
        }
        let path = entry.path();
        if !path.starts_with(&base) {
            continue;
        }
        let relative_depth = path
            .strip_prefix(&base)
            .map(|path| path.components().count())
            .unwrap_or(0);
        if relative_depth > depth {
            continue;
        }
        if entry.file_type().is_some_and(|kind| kind.is_symlink()) {
            continue;
        }
        let Some(display) = relative_string(&ctx.root, path) else {
            continue;
        };
        lines.push(if entry.file_type().is_some_and(|kind| kind.is_dir()) {
            format!("{display}/")
        } else {
            display
        });
        if lines.len() >= MAX_LISTING_ENTRIES {
            truncated = true;
            break;
        }
    }
    lines.sort();
    let mut output = lines.join("\n");
    if truncated {
        output.push_str("\n...[listing truncated]");
    }
    if output.is_empty() {
        output = "(empty)".to_owned();
    }
    Ok(output)
}

pub(super) fn read_file(ctx: &ToolContext, arguments: &Value) -> Result<String> {
    let relative = require_string(arguments, "path")?;
    let read = ctx
        .read_bounded(&relative, ctx.max_read_bytes)
        .with_context(|| format!("failed to read {relative}"))?;
    if read.bytes.contains(&0) {
        bail!("binary file cannot be read as text: {relative}");
    }
    let usable = match std::str::from_utf8(&read.bytes) {
        Ok(_) => &read.bytes[..],
        Err(error) if read.truncated => &read.bytes[..error.valid_up_to()],
        Err(_) => bail!("file is not valid UTF-8 text: {relative}"),
    };
    let text = std::str::from_utf8(usable).expect("validated UTF-8");
    let total_lines = text.lines().count();
    let start = optional_u64(arguments, "start_line")?.unwrap_or(1).max(1) as usize;
    let end = optional_u64(arguments, "end_line")?
        .map(|value| value as usize)
        .unwrap_or(total_lines)
        .max(start);
    if start > total_lines && total_lines > 0 {
        bail!("start_line {start} is beyond the end of {relative} ({total_lines} lines)");
    }
    let selected = if start > 1 || end < total_lines {
        let mut selected: Vec<&str> = text
            .lines()
            .skip(start.saturating_sub(1))
            .take(end.saturating_sub(start) + 1)
            .collect();
        if !selected.is_empty() {
            selected.push("");
        }
        selected.join("\n")
    } else {
        text.to_owned()
    };
    let header = format!(
        "path: {relative}\nhash: {}\nsize: {} bytes\nlines: {}-{} of {total_lines}{}\n\n",
        read.hash,
        read.size,
        start.min(total_lines.max(1)),
        end.min(total_lines.max(1)),
        if read.truncated {
            "\nnote: file content truncated at the configured read limit"
        } else {
            ""
        }
    );
    Ok(format!("{header}{selected}"))
}

pub(super) fn search(ctx: &ToolContext, arguments: &Value) -> Result<String> {
    let query = require_string(arguments, "query")?;
    if query.is_empty() {
        bail!("query must not be empty");
    }
    if query.len() > 4096 {
        bail!("query is too long");
    }
    let regex = arguments
        .get("regex")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let pattern = if regex {
        query.clone()
    } else {
        regex::escape(&query)
    };
    let expression = regex::RegexBuilder::new(&pattern)
        .size_limit(1024 * 1024)
        .dfa_size_limit(1024 * 1024)
        .build()
        .with_context(|| "invalid regular expression")?;
    let max_results = optional_u64(arguments, "max_results")?
        .unwrap_or(40)
        .clamp(1, 100) as usize;
    let relative = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
    let base = validate_search_root(ctx, relative)?;
    let max_file_size = ctx.max_read_bytes.saturating_mul(2);
    let mut matches = Vec::new();
    let mut skipped_large = 0usize;
    let mut skipped_binary = 0usize;
    let mut scanned = 0usize;

    for entry in walker(&ctx.root)? {
        let Ok(entry) = entry else { continue };
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        let path = entry.path();
        if !path.starts_with(&base) {
            continue;
        }
        scanned += 1;
        if scanned > MAX_SEARCH_FILES {
            bail!("search exceeded the {MAX_SEARCH_FILES}-file traversal limit");
        }
        let Some(relative_path) = relative_string(&ctx.root, path) else {
            continue;
        };
        let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if size > max_file_size as u64 {
            skipped_large += 1;
            continue;
        }
        let Ok(read) = ctx.read_bounded(&relative_path, max_file_size) else {
            continue;
        };
        if read.truncated {
            skipped_large += 1;
            continue;
        }
        if read.bytes.contains(&0) {
            skipped_binary += 1;
            continue;
        }
        let Ok(text) = String::from_utf8(read.bytes) else {
            skipped_binary += 1;
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            if expression.is_match(line) {
                matches.push(format!(
                    "{relative_path}:{}: {}",
                    index + 1,
                    truncate_bytes(line.trim(), 300)
                ));
                if matches.len() >= max_results {
                    break;
                }
            }
        }
        if matches.len() >= max_results {
            break;
        }
    }

    if matches.is_empty() {
        let mut output = "no matches".to_owned();
        if skipped_large > 0 || skipped_binary > 0 {
            output.push_str(&format!(
                "\nskipped {skipped_large} large files and {skipped_binary} binary files"
            ));
        }
        return Ok(output);
    }
    let mut output = matches.join("\n");
    if matches.len() >= max_results {
        output.push_str("\n...[search truncated]");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::tools::{ToolContext, ToolRole};
    use serde_json::json;

    #[test]
    fn reads_bounded_lines_and_returns_hash() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "one\ntwo\nthree\n").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024, 1024).unwrap();
        let result = context.execute(
            ToolRole::Planner,
            "read_file",
            &json!({"path": "a.txt", "start_line": 2, "end_line": 2}),
        );
        assert!(!result.is_error);
        assert!(result.content.contains("hash: "));
        assert!(result.content.contains("two"));
        assert!(!result.content.contains("one"));
    }

    #[test]
    fn rejects_binary_and_large_output() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("bin"), [0u8, 1, 2]).unwrap();
        let context = ToolContext::new(directory.path(), None, 1024, 1024).unwrap();
        let result = context.execute(ToolRole::Planner, "read_file", &json!({"path": "bin"}));
        assert!(result.is_error);
        assert!(result.content.contains("binary"));
    }

    #[test]
    fn searches_literal_and_regex() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "alpha\nbeta 42\n").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let literal = context.execute(ToolRole::Planner, "search", &json!({"query": "beta"}));
        assert!(literal.content.contains("a.txt:2"));
        let regex = context.execute(
            ToolRole::Planner,
            "search",
            &json!({"query": "\\d+", "regex": true}),
        );
        assert!(regex.content.contains("beta 42"));
    }

    #[test]
    fn subdirectory_search_keeps_root_relative_exclusions() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("sub")).unwrap();
        std::fs::write(directory.path().join("sub/a.txt"), "needle\n").unwrap();
        std::fs::write(directory.path().join("sub/.env"), "needle\n").unwrap();
        std::fs::write(directory.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(directory.path().join("sub/ignored.txt"), "needle\n").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let result = context.execute(
            ToolRole::Planner,
            "search",
            &json!({"query": "needle", "path": "sub"}),
        );
        assert!(result.content.contains("sub/a.txt"));
        assert!(!result.content.contains(".env"));
        assert!(!result.content.contains("ignored.txt"));
    }
}
