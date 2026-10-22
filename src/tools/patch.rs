use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use super::{ToolContext, normalize_relative, optional_bool, require_string};
use crate::util::hash_bytes;

#[derive(Debug)]
enum Action {
    Create,
    Edit,
    Delete,
}

#[derive(Debug)]
struct PlannedEdit {
    relative: String,
    action: Action,
    new_bytes: Vec<u8>,
    preimage: Option<Vec<u8>>,
    mode: Option<u32>,
    expected_hash: Option<String>,
}

pub(super) fn apply_patch(ctx: &ToolContext, arguments: &Value) -> Result<String> {
    let edits = arguments
        .get("edits")
        .and_then(Value::as_array)
        .context("missing edits array")?;
    if edits.is_empty() {
        bail!("edits must not be empty");
    }
    if edits.len() > 20 {
        bail!("at most 20 edits are allowed per call");
    }
    let mut seen = BTreeSet::new();
    let mut validated = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        let relative = require_string(edit, "path")
            .with_context(|| format!("edit {index} has an invalid path"))?;
        if !ctx.path_is_writable(&relative) {
            bail!("edit {index} ({relative}) is outside the approved plan file scope");
        }
        let normalized = normalize_relative(&relative);
        if normalized.is_empty() {
            bail!("edit {index} has an empty path");
        }
        if !seen.insert(normalized) {
            bail!("duplicate path in one apply_patch call: {relative}");
        }
        validated.push(relative);
    }
    let mut planned = Vec::with_capacity(edits.len());
    for (index, (relative, edit)) in validated.iter().zip(edits).enumerate() {
        let planned_edit = plan_edit(ctx, relative, edit)
            .with_context(|| format!("edit {index} ({relative}) failed validation"))?;
        planned.push(planned_edit);
    }

    let mut output = Vec::with_capacity(planned.len());
    let mut applied: Vec<usize> = Vec::new();
    for (index, edit) in planned.iter().enumerate() {
        let result = apply_one(ctx, edit);
        match result {
            Ok(line) => {
                applied.push(index);
                output.push(line);
            }
            Err(error) => {
                let mut rollback_errors = Vec::new();
                for applied_index in applied.iter().rev() {
                    if let Err(rollback_error) = restore(ctx, &planned[*applied_index]) {
                        rollback_errors.push(format!(
                            "{}: {rollback_error:#}",
                            planned[*applied_index].relative
                        ));
                    }
                }
                let message = format!("{error:#}");
                if rollback_errors.is_empty() {
                    bail!(
                        "{message}: apply_patch aborted and {} earlier edit(s) were restored",
                        applied.len()
                    );
                }
                bail!(
                    "{message}: apply_patch aborted with incomplete rollback:\n{}",
                    rollback_errors.join("\n")
                );
            }
        }
    }
    Ok(output.join("\n"))
}

fn apply_one(ctx: &ToolContext, edit: &PlannedEdit) -> Result<String> {
    verify_preimage(ctx, edit)?;
    match edit.action {
        Action::Create => {
            if ctx.exists(&edit.relative) {
                bail!("file appeared while applying create: {}", edit.relative);
            }
            ctx.write_file(&edit.relative, &edit.new_bytes, edit.mode)?;
            Ok(format!(
                "{}: created ({} bytes, hash {})",
                edit.relative,
                edit.new_bytes.len(),
                hash_bytes(&edit.new_bytes)
            ))
        }
        Action::Edit => {
            ctx.write_file(&edit.relative, &edit.new_bytes, edit.mode)?;
            Ok(format!(
                "{}: updated ({} bytes, hash {})",
                edit.relative,
                edit.new_bytes.len(),
                hash_bytes(&edit.new_bytes)
            ))
        }
        Action::Delete => {
            ctx.remove_file(&edit.relative)?;
            Ok(format!("{}: deleted", edit.relative))
        }
    }
}

/// Best-effort restoration with the exact preimage captured during planning.
/// Errors are returned, never swallowed.
fn restore(ctx: &ToolContext, edit: &PlannedEdit) -> Result<()> {
    match edit.action {
        Action::Create => {
            if ctx.exists(&edit.relative) {
                ctx.remove_file(&edit.relative)?;
            }
            Ok(())
        }
        Action::Edit | Action::Delete => {
            let preimage = edit
                .preimage
                .as_deref()
                .context("internal error: missing preimage for rollback")?;
            ctx.write_file(&edit.relative, preimage, edit.mode)
        }
    }
}

fn verify_preimage(ctx: &ToolContext, edit: &PlannedEdit) -> Result<()> {
    let Some(expected) = edit.expected_hash.as_deref() else {
        return Ok(());
    };
    let actual = ctx
        .read_bounded(&edit.relative, 1)
        .with_context(|| format!("failed to verify {}", edit.relative))?
        .hash;
    if actual != expected {
        bail!("file changed while applying edit: {}", edit.relative);
    }
    Ok(())
}

fn plan_edit(ctx: &ToolContext, relative: &str, edit: &Value) -> Result<PlannedEdit> {
    let create = optional_bool(edit, "create")?;
    let delete = optional_bool(edit, "delete")?;
    if create && delete {
        bail!("create and delete cannot both be true");
    }
    let expected_hash = match edit.get("expected_hash") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => bail!("expected_hash must be a string or null"),
    };
    if let Some(hash) = &expected_hash
        && (hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        bail!("expected_hash must be a BLAKE3 hex digest");
    }
    let old_text = match edit.get("old_text") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => bail!("old_text must be a string or null"),
    };
    let new_text = match edit.get("new_text") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => bail!("new_text must be a string or null"),
    };

    if create {
        if expected_hash.is_some() {
            bail!("create must use expected_hash = null");
        }
        let bytes = new_text.context("create requires new_text")?.into_bytes();
        if ctx.exists(relative) {
            bail!("file already exists");
        }
        check_new_size(ctx, &bytes)?;
        return Ok(PlannedEdit {
            relative: relative.to_owned(),
            action: Action::Create,
            new_bytes: bytes,
            preimage: None,
            mode: Some(0o644),
            expected_hash: None,
        });
    }

    let expected_hash = expected_hash.context("edits and deletes require expected_hash")?;
    let limit = edit_limit(ctx);
    let read = ctx
        .read_bounded(relative, limit)
        .with_context(|| format!("failed to read {relative}"))?;
    if read.hash != expected_hash {
        bail!("preimage hash mismatch; read the file again before editing");
    }
    let mode = ctx.mode_of(relative);
    if delete {
        if old_text.is_some() || new_text.is_some() {
            bail!("delete must not include old_text or new_text");
        }
        return Ok(PlannedEdit {
            relative: relative.to_owned(),
            action: Action::Delete,
            new_bytes: Vec::new(),
            preimage: Some(read.bytes),
            mode,
            expected_hash: Some(expected_hash),
        });
    }

    let new_text = new_text.context("edit requires new_text")?;
    let replacement = match old_text {
        Some(old_text) => {
            if old_text.is_empty() {
                bail!("old_text must not be empty; omit it to replace the whole file");
            }
            if read.truncated {
                bail!(
                    "{relative} exceeds the edit size limit; replace the whole file instead of matching old_text"
                );
            }
            let text = std::str::from_utf8(&read.bytes)
                .context("text edits require a UTF-8 file; omit old_text to replace it")?;
            let occurrences = text.matches(&old_text).count();
            if occurrences != 1 {
                bail!(
                    "old_text must occur exactly once in {relative}, found {occurrences} occurrences"
                );
            }
            text.replacen(&old_text, &new_text, 1).into_bytes()
        }
        None => new_text.into_bytes(),
    };
    check_new_size(ctx, &replacement)?;
    Ok(PlannedEdit {
        relative: relative.to_owned(),
        action: Action::Edit,
        new_bytes: replacement,
        preimage: Some(read.bytes),
        mode,
        expected_hash: Some(expected_hash),
    })
}

fn edit_limit(ctx: &ToolContext) -> usize {
    ctx.max_read_bytes.saturating_mul(8).max(1024 * 1024)
}

fn check_new_size(ctx: &ToolContext, bytes: &[u8]) -> Result<()> {
    let limit = edit_limit(ctx);
    if bytes.len() > limit {
        bail!(
            "replacement is too large ({} bytes, limit {limit})",
            bytes.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::repo_fs::hash_file_streaming;
    use crate::tools::{ToolContext, ToolRole};
    use serde_json::json;

    #[test]
    fn requires_matching_hash_and_unique_old_text() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "hello world\n").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let hash = hash_file_streaming(&directory.path().join("a.txt")).unwrap();
        let success = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{
                "path": "a.txt",
                "expected_hash": hash,
                "old_text": "world",
                "new_text": "there"
            }]}),
        );
        assert!(!success.is_error, "{}", success.content);
        assert_eq!(
            std::fs::read_to_string(directory.path().join("a.txt")).unwrap(),
            "hello there\n"
        );

        let stale = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{
                "path": "a.txt",
                "expected_hash": "0".repeat(64),
                "old_text": "there",
                "new_text": "x"
            }]}),
        );
        assert!(stale.is_error);
        assert!(stale.content.contains("preimage hash mismatch"));
    }

    #[test]
    fn creates_nested_files_and_rejects_absolute_paths() {
        let directory = tempfile::tempdir().unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let result = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{
                "path": "src/new.rs",
                "create": true,
                "new_text": "fn main() {}\n"
            }]}),
        );
        assert!(!result.is_error, "{}", result.content);
        assert!(directory.path().join("src/new.rs").exists());

        let absolute = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{
                "path": "/tmp/escape",
                "create": true,
                "new_text": "x"
            }]}),
        );
        assert!(absolute.is_error);
    }

    #[test]
    fn deletes_only_with_matching_hash() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "a").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let hash = hash_file_streaming(&directory.path().join("a.txt")).unwrap();
        let result = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{"path": "a.txt", "expected_hash": hash, "delete": true}]}),
        );
        assert!(!result.is_error, "{}", result.content);
        assert!(!directory.path().join("a.txt").exists());
    }

    #[test]
    fn rejects_duplicate_normalized_paths() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "a").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let result = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [
                {"path": "a.txt", "create": true, "new_text": "x"},
                {"path": "./a.txt", "create": true, "new_text": "y"}
            ]}),
        );
        assert!(result.is_error);
        assert!(result.content.contains("duplicate path"));
    }

    #[test]
    fn rejects_writes_outside_approved_scope() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("approved.txt"), "a").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024)
            .unwrap()
            .with_scope(vec!["approved.txt".to_owned()]);
        let allowed = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{"path": "approved.txt", "expected_hash": hash_file_streaming(&directory.path().join("approved.txt")).unwrap(), "new_text": "b"}]}),
        );
        assert!(!allowed.is_error, "{}", allowed.content);
        let denied = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [{"path": "other.txt", "create": true, "new_text": "x"}]}),
        );
        assert!(denied.is_error);
        assert!(denied.content.contains("outside the approved plan"));
        assert!(!directory.path().join("other.txt").exists());
    }

    #[test]
    fn restores_earlier_edits_when_a_later_edit_fails() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "a").unwrap();
        std::fs::write(directory.path().join("b.txt"), "b").unwrap();
        let context = ToolContext::new(directory.path(), None, 1024 * 1024, 1024 * 1024).unwrap();
        let hash_a = hash_file_streaming(&directory.path().join("a.txt")).unwrap();
        let result = context.execute(
            ToolRole::Executor,
            "apply_patch",
            &json!({"edits": [
                {"path": "a.txt", "expected_hash": hash_a, "new_text": "changed"},
                {"path": "b.txt", "expected_hash": "0".repeat(64), "new_text": "changed"}
            ]}),
        );
        assert!(result.is_error);
        assert_eq!(
            std::fs::read_to_string(directory.path().join("a.txt")).unwrap(),
            "a"
        );
        assert_eq!(
            std::fs::read_to_string(directory.path().join("b.txt")).unwrap(),
            "b"
        );
    }
}
