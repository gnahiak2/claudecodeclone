use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub fn hash_file(path: &Path) -> Result<String> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to hash {}", path.display()))?;
    Ok(hash_bytes(&bytes))
}

pub fn redact(text: &str, secrets: &[String]) -> String {
    let mut ordered: Vec<&String> = secrets.iter().filter(|secret| !secret.is_empty()).collect();
    // Longest first so a secret that contains another secret cannot leak its tail.
    ordered.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
    let mut output = text.to_owned();
    for secret in ordered {
        output = output.replace(secret.as_str(), "[REDACTED]");
    }
    output
}

/// Escape control characters for terminal display. Patch bytes and file
/// contents are never passed through this; it is only for human-facing output.
pub fn sanitize_terminal(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' | '\t' => output.push(character),
            '\r' => output.push_str("\\r"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{{{:04x}}}", character as u32));
            }
            character => output.push(character),
        }
    }
    output
}

pub fn truncate_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }
    let suffix = "\n...[output truncated]";
    let (budget, suffix) = if max_bytes > suffix.len() {
        (max_bytes - suffix.len(), suffix)
    } else {
        (max_bytes, "")
    };
    let mut end = budget.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut output = text[..end].to_owned();
    output.push_str(suffix);
    output
}

pub fn atomic_write(path: &Path, data: &[u8], mode: Option<u32>) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("failed to create a temporary file in {}", parent.display()))?;
    temp.write_all(data)
        .with_context(|| format!("failed to write temporary data for {}", path.display()))?;
    temp.as_file().sync_all()?;
    if let Some(mode) = mode {
        #[cfg(unix)]
        {
            temp.as_file()
                .set_permissions(std::fs::Permissions::from_mode(mode))?
        }
        #[cfg(not(unix))]
        let _ = mode;
    }
    temp.persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("failed to atomically replace {}", path.display()))?;
    Ok(())
}

pub fn file_mode(path: &Path) -> Option<u32> {
    #[cfg(unix)]
    {
        std::fs::symlink_metadata(path)
            .ok()
            .map(|metadata| metadata.permissions().mode() & 0o777)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        None
    }
}

pub fn is_sensitive_rel(path: &Path) -> bool {
    let components: Vec<String> = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(|value| value.to_ascii_lowercase()),
            _ => None,
        })
        .collect();

    const SENSITIVE_DIRS: &[&str] = &[
        ".git",
        ".hg",
        ".svn",
        ".aws",
        ".ssh",
        ".gnupg",
        ".commandcode",
        "__pycache__",
    ];
    if components
        .iter()
        .any(|component| SENSITIVE_DIRS.contains(&component.as_str()))
    {
        return true;
    }
    if components
        .windows(2)
        .any(|pair| pair[0] == ".config" && pair[1] == "gcloud")
    {
        return true;
    }
    let Some(name) = components.last() else {
        return false;
    };
    if name.starts_with(".env") {
        return true;
    }
    if matches!(
        name.as_str(),
        ".netrc"
            | ".npmrc"
            | ".pypirc"
            | "credentials"
            | "credentials.json"
            | "secrets.json"
            | "secrets.toml"
            | "id_rsa"
            | "id_ed25519"
    ) {
        return true;
    }
    let extension = Path::new(name)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    matches!(extension, "pem" | "key" | "p12" | "pfx" | "jks")
}

pub fn is_ignored_dir(name: &str) -> bool {
    matches!(
        name,
        "target"
            | "node_modules"
            | "dist"
            | "coverage"
            | ".next"
            | ".venv"
            | "venv"
            | "__pycache__"
    )
}

pub fn safe_join(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    if path.is_absolute() {
        bail!("path must be relative");
    }
    let mut depth = 0usize;
    for component in path.components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir => bail!("path must not contain '..'"),
            Component::RootDir | Component::Prefix(_) => bail!("path must be relative"),
        }
    }
    if depth == 0 {
        bail!("path must name a file");
    }
    if is_sensitive_rel(path) {
        bail!("path is excluded because it may contain secrets");
    }
    Ok(root.join(path))
}

pub fn ensure_contained(root: &Path, candidate: &Path) -> Result<()> {
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("failed to resolve root {}", root.display()))?;
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", candidate.display()))?;
    if !canonical.starts_with(&canonical_root) {
        bail!("path escapes the workspace");
    }
    Ok(())
}

pub fn display_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redaction_replaces_every_secret() {
        let text = "key=abc and again abc";
        assert_eq!(
            redact(text, &["abc".to_owned()]),
            "key=[REDACTED] and again [REDACTED]"
        );
    }

    #[test]
    fn truncation_respects_utf8_boundaries() {
        let text = "éééééééééé";
        let result = truncate_bytes(text, 12);
        assert!(result.len() <= 12);
        assert!(result.is_char_boundary(result.len()));
    }

    #[test]
    fn detects_sensitive_paths() {
        for path in [
            ".env",
            "nested/.env.local",
            ".ssh/config",
            "certs/server.pem",
            ".commandcode/taste/taste.md",
            "secrets.toml",
        ] {
            assert!(is_sensitive_rel(Path::new(path)), "{path}");
        }
        for path in ["src/main.rs", "README.md", ".github/workflows/ci.yml"] {
            assert!(!is_sensitive_rel(Path::new(path)), "{path}");
        }
    }

    #[test]
    fn safe_join_rejects_traversal_and_escape() {
        let root = Path::new("/tmp/workspace");
        assert!(safe_join(root, "../secret").is_err());
        assert!(safe_join(root, "/etc/passwd").is_err());
        assert!(safe_join(root, "").is_err());
        assert!(safe_join(root, "src/main.rs").is_ok());
    }
}
