use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};

use crate::util::{file_mode, is_ignored_dir, is_sensitive_rel};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    pub hash: String,
    pub size: u64,
    pub mode: Option<u32>,
}

/// Authoritative inventory used for baseline manifests, staged diffs, evidence
/// and apply. It deliberately does not consult `.gitignore` or other editable
/// ignore rules, so changing them cannot hide files from review or tests.
pub fn collect_files(root: &Path) -> Result<BTreeMap<String, FileEntry>> {
    let mut files = BTreeMap::new();
    if !root.exists() {
        return Ok(files);
    }
    let builder = exact_walker(root);
    for entry in builder {
        let entry = entry.with_context(|| format!("failed to walk {}", root.display()))?;
        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        let relative = match path.strip_prefix(root) {
            Ok(relative) => relative,
            Err(_) => continue,
        };
        if is_sensitive_rel(relative) {
            continue;
        }
        let Some(relative) = relative.to_str() else {
            continue;
        };
        let hash = hash_file_streaming(path)?;
        let metadata = entry
            .metadata()
            .with_context(|| format!("failed to read metadata for {}", path.display()))?;
        files.insert(
            relative.replace('\\', "/"),
            FileEntry {
                hash,
                size: metadata.len(),
                mode: file_mode(path),
            },
        );
    }
    Ok(files)
}

fn exact_walker(root: &Path) -> ignore::Walk {
    let root = root.to_path_buf();
    let filter_root = root.clone();
    let mut builder = WalkBuilder::new(&root);
    builder
        .hidden(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .ignore(false)
        .require_git(false)
        .follow_links(false)
        .filter_entry(move |entry| {
            let name = entry.file_name().to_string_lossy();
            if entry.depth() > 0 && (name == ".git" || is_ignored_dir(&name)) {
                return false;
            }
            let relative = entry
                .path()
                .strip_prefix(&filter_root)
                .unwrap_or(entry.path());
            !is_sensitive_rel(relative)
        });
    builder.build()
}

pub fn walker(root: &Path) -> Result<ignore::Walk> {
    let root = root.to_path_buf();
    let filter_root = root.clone();
    let mut builder = WalkBuilder::new(&root);
    builder
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .require_git(false)
        .follow_links(false)
        .filter_entry(move |entry| {
            let name = entry.file_name().to_string_lossy();
            if entry.depth() > 0 && (name == ".git" || is_ignored_dir(&name)) {
                return false;
            }
            let relative = entry
                .path()
                .strip_prefix(&filter_root)
                .unwrap_or(entry.path());
            !is_sensitive_rel(relative)
        });
    Ok(builder.build())
}

pub fn hash_file_streaming(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn relative_string(root: &Path, path: &Path) -> Option<String> {
    path.strip_prefix(root)
        .ok()
        .and_then(Path::to_str)
        .map(|value| value.replace('\\', "/"))
}

pub fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(source)
        .with_context(|| format!("failed to inspect {}", source.display()))?;
    if metadata.file_type().is_symlink() {
        anyhow::bail!("refusing to copy symlink {}", source.display());
    }
    if !metadata.is_file() {
        anyhow::bail!("refusing to copy non-file {}", source.display());
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::copy(source, destination).with_context(|| {
        format!(
            "failed to copy {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    let source_hash = hash_file_streaming(source)?;
    let destination_hash = hash_file_streaming(destination)?;
    if source_hash != destination_hash {
        anyhow::bail!(
            "source changed while snapshotting {}; refusing an inconsistent baseline",
            source.display()
        );
    }
    if let Some(mode) = file_mode(source) {
        #[cfg(unix)]
        std::fs::set_permissions(
            destination,
            std::os::unix::fs::PermissionsExt::from_mode(mode),
        )?;
    }
    Ok(())
}

pub fn safe_relative_path(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        anyhow::bail!("path must be a relative path without '..'");
    }
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_files_and_skips_secrets() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "a").unwrap();
        std::fs::write(directory.path().join(".env"), "SECRET=1").unwrap();
        std::fs::create_dir(directory.path().join(".ssh")).unwrap();
        std::fs::write(directory.path().join(".ssh/config"), "secret").unwrap();
        let files = collect_files(directory.path()).unwrap();
        assert!(files.contains_key("a.txt"));
        assert!(!files.contains_key(".env"));
        assert!(!files.keys().any(|path| path.starts_with(".ssh")));
    }
}
