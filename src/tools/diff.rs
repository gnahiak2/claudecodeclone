use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ToolContext;
use crate::repo_fs::collect_files;
use crate::util::truncate_bytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeStatus {
    Added,
    Modified,
    Deleted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangedFile {
    pub path: String,
    pub status: ChangeStatus,
    pub old_hash: Option<String>,
    pub new_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffReport {
    pub changed: Vec<ChangedFile>,
    pub patch: String,
    pub truncated: bool,
}

pub fn compute_diff(
    stage_root: &Path,
    baseline_root: &Path,
    max_bytes: usize,
) -> Result<DiffReport> {
    let stage = collect_files(stage_root)?;
    let baseline = collect_files(baseline_root)?;
    let paths: BTreeSet<&String> = stage.keys().chain(baseline.keys()).collect();
    let mut changed = Vec::new();
    let mut patch = String::new();
    let mut truncated = false;

    for path in paths {
        let before = baseline.get(path);
        let after = stage.get(path);
        let status = match (before, after) {
            (None, Some(_)) => ChangeStatus::Added,
            (Some(_), None) => ChangeStatus::Deleted,
            (Some(before), Some(after))
                if before.hash != after.hash || before.mode != after.mode =>
            {
                ChangeStatus::Modified
            }
            _ => continue,
        };
        changed.push(ChangedFile {
            path: path.clone(),
            status,
            old_hash: before.map(|entry| entry.hash.clone()),
            new_hash: after.map(|entry| entry.hash.clone()),
        });

        if truncated {
            continue;
        }
        let old_path = baseline_root.join(path);
        let new_path = stage_root.join(path);
        let old = if before.is_some() {
            read_capped(&old_path)?
        } else {
            Capped::Empty
        };
        let new = if after.is_some() {
            read_capped(&new_path)?
        } else {
            Capped::Empty
        };
        if let Some(size) = old.too_large().or_else(|| new.too_large()) {
            let section =
                format!("File {path} is too large to display as a text diff ({size} bytes)\n");
            if patch.len() + section.len() > max_bytes {
                truncated = true;
            } else {
                patch.push_str(&section);
            }
            continue;
        }
        let old_bytes = old.into_bytes();
        let new_bytes = new.into_bytes();
        let section = if old_bytes.contains(&0)
            || new_bytes.contains(&0)
            || std::str::from_utf8(&old_bytes).is_err()
            || std::str::from_utf8(&new_bytes).is_err()
        {
            format!("Binary files a/{path} and b/{path} differ\n")
        } else {
            let old_text = String::from_utf8(old_bytes).unwrap_or_default();
            let new_text = String::from_utf8(new_bytes).unwrap_or_default();
            let generated = diffy::create_patch(&old_text, &new_text).to_string();
            let body = generated
                .find("@@")
                .map(|index| &generated[index..])
                .unwrap_or(generated.as_str());
            if body.trim().is_empty() {
                format!("--- a/{path}\n+++ b/{path}\n(file contents differ)\n")
            } else {
                format!("--- a/{path}\n+++ b/{path}\n{body}\n")
            }
        };
        if patch.len() + section.len() > max_bytes {
            let remaining = max_bytes.saturating_sub(patch.len());
            patch.push_str(&truncate_bytes(&section, remaining));
            truncated = true;
        } else {
            patch.push_str(&section);
        }
    }

    if changed.is_empty() {
        patch.push_str("no changes\n");
    }
    if truncated {
        patch.push_str("\n...[diff truncated]\n");
    }
    Ok(DiffReport {
        changed,
        patch,
        truncated,
    })
}

const MAX_DIFF_FILE_BYTES: u64 = 8 * 1024 * 1024;

enum Capped {
    Bytes(Vec<u8>),
    TooLarge(u64),
    Empty,
}

impl Capped {
    fn into_bytes(self) -> Vec<u8> {
        match self {
            Self::Bytes(bytes) => bytes,
            Self::TooLarge(_) | Self::Empty => Vec::new(),
        }
    }

    fn too_large(&self) -> Option<u64> {
        match self {
            Self::TooLarge(size) => Some(*size),
            _ => None,
        }
    }
}

fn read_capped(path: &Path) -> Result<Capped> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        anyhow::bail!("refusing to diff a non-regular file: {}", path.display());
    }
    if metadata.len() > MAX_DIFF_FILE_BYTES {
        return Ok(Capped::TooLarge(metadata.len()));
    }
    let mut file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut bytes = Vec::new();
    file.by_ref()
        .take(MAX_DIFF_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("failed to read {}", path.display()))?;
    if bytes.len() as u64 > MAX_DIFF_FILE_BYTES {
        return Ok(Capped::TooLarge(bytes.len() as u64));
    }
    Ok(Capped::Bytes(bytes))
}

pub(super) fn tool_diff(ctx: &ToolContext, _arguments: &Value) -> Result<String> {
    let baseline = ctx
        .baseline
        .as_deref()
        .context("diff is unavailable without a baseline snapshot")?;
    let report = compute_diff(&ctx.root, baseline, ctx.max_output_bytes)?;
    let mut output = String::new();
    for file in &report.changed {
        output.push_str(&format!("{:?} {}\n", file.status, file.path));
    }
    if output.is_empty() {
        return Ok("no changes".to_owned());
    }
    output.push('\n');
    output.push_str(&report.patch);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_added_modified_deleted_and_patch() {
        let baseline = tempfile::tempdir().unwrap();
        let stage = tempfile::tempdir().unwrap();
        std::fs::write(baseline.path().join("keep.txt"), "same\n").unwrap();
        std::fs::write(stage.path().join("keep.txt"), "same\n").unwrap();
        std::fs::write(baseline.path().join("change.txt"), "old\n").unwrap();
        std::fs::write(stage.path().join("change.txt"), "new\n").unwrap();
        std::fs::write(baseline.path().join("delete.txt"), "gone\n").unwrap();
        std::fs::write(stage.path().join("add.txt"), "added\n").unwrap();

        let report = compute_diff(stage.path(), baseline.path(), 1024 * 1024).unwrap();
        assert_eq!(report.changed.len(), 3);
        assert!(
            report
                .changed
                .iter()
                .any(|file| file.path == "add.txt" && file.status == ChangeStatus::Added)
        );
        assert!(
            report
                .changed
                .iter()
                .any(|file| file.path == "change.txt" && file.status == ChangeStatus::Modified)
        );
        assert!(
            report
                .changed
                .iter()
                .any(|file| file.path == "delete.txt" && file.status == ChangeStatus::Deleted)
        );
        assert!(report.patch.contains("+new"));
        assert!(report.patch.contains("-old"));
    }

    #[test]
    fn marks_binary_changes() {
        let baseline = tempfile::tempdir().unwrap();
        let stage = tempfile::tempdir().unwrap();
        std::fs::write(baseline.path().join("a.bin"), [0u8, 1]).unwrap();
        std::fs::write(stage.path().join("a.bin"), [0u8, 2]).unwrap();
        let report = compute_diff(stage.path(), baseline.path(), 1024 * 1024).unwrap();
        assert!(report.patch.contains("Binary files"));
    }
}
