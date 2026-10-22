use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::repo_fs::{FileEntry, collect_files, relative_string, safe_relative_path, walker};
use crate::safe_fs::DirHandle;
use crate::tools::{ChangeStatus, ChangedFile, DiffReport, ToolContext, compute_diff};
use crate::util::{atomic_write, file_mode, hash_bytes, is_sensitive_rel};

const MAX_APPLY_FILE_BYTES: usize = 256 * 1024 * 1024;

const MAX_FILES: usize = 20_000;
const MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
const MAX_WARNINGS: usize = 20;

/// The reviewed, evaluated snapshot that an apply is authorized to write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyAuthorization {
    pub snapshot_hash: String,
    pub evaluation_id: String,
    pub approved_at_ms: u64,
}

/// Evidence produced by the deterministic evaluator for one exact snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyEvidence {
    pub id: String,
    pub snapshot_hash: String,
    pub policy_hash: String,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyOperation {
    pub file: ChangedFile,
    pub state: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceState {
    pub project_root: PathBuf,
    pub stage_root: PathBuf,
    pub baseline_root: PathBuf,
    pub files: BTreeMap<String, FileEntry>,
}

#[derive(Debug, Clone)]
pub struct Workspace {
    state: WorkspaceState,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyJournal {
    pub id: String,
    pub project_root: PathBuf,
    pub backup_dir: PathBuf,
    pub evaluation_id: String,
    pub snapshot_hash: String,
    pub state: String,
    pub operations: Vec<ApplyOperation>,
    pub rollback_errors: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyReport {
    pub applied: Vec<ChangedFile>,
    pub backup_dir: PathBuf,
    pub journal: PathBuf,
}

impl Workspace {
    pub fn create(project_root: &Path, run_dir: &Path) -> Result<Self> {
        let project_root = project_root.canonicalize().with_context(|| {
            format!("failed to resolve project root {}", project_root.display())
        })?;
        create_private_dir(run_dir)?;
        let run_dir = run_dir
            .canonicalize()
            .with_context(|| format!("failed to resolve run directory {}", run_dir.display()))?;
        if run_dir.starts_with(&project_root) {
            bail!(
                "the run directory must be outside the project; refusing to snapshot {}",
                run_dir.display()
            );
        }
        let stage_root = run_dir.join("workspace");
        let baseline_root = run_dir.join("baseline");
        if stage_root.exists() || baseline_root.exists() {
            bail!("run directory already contains a workspace snapshot");
        }
        create_private_dir(&stage_root)?;
        create_private_dir(&baseline_root)?;

        let (paths, mut warnings) = source_paths(&project_root)?;
        if paths.len() > MAX_FILES {
            bail!("repository exceeds the {MAX_FILES} file snapshot limit");
        }
        let source_handle = DirHandle::open_root(&project_root)?;
        let mut total_bytes = 0u64;
        for relative in &paths {
            let relative_path = safe_relative_path(relative)?;
            if is_sensitive_rel(&relative_path) {
                continue;
            }
            let metadata = source_handle.metadata(relative).with_context(|| {
                format!("source {relative} disappeared or became unreadable while snapshotting")
            })?;
            if metadata.is_symlink {
                warnings.push(format!("skipped symlink {relative}"));
                if warnings.len() > MAX_WARNINGS {
                    bail!(
                        "too many unreadable or skipped entries ({MAX_WARNINGS}); refusing an incomplete baseline"
                    );
                }
                continue;
            }
            if !metadata.is_file {
                continue;
            }
            total_bytes = total_bytes.saturating_add(metadata.size);
            if total_bytes > MAX_TOTAL_BYTES {
                bail!(
                    "repository snapshot exceeds the {} MiB safety limit",
                    MAX_TOTAL_BYTES / (1024 * 1024)
                );
            }
            let read = source_handle
                .read(relative, MAX_APPLY_FILE_BYTES)
                .with_context(|| format!("failed to read source file {relative}"))?;
            if read.truncated {
                bail!(
                    "source file {relative} is too large to snapshot safely (limit {} bytes)",
                    crate::util::display_bytes(MAX_APPLY_FILE_BYTES)
                );
            }
            let baseline_path = baseline_root.join(&relative_path);
            if let Some(parent) = baseline_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            atomic_write(&baseline_path, &read.bytes, Some(metadata.mode.max(0o400)))?;
            let stage_path = stage_root.join(&relative_path);
            if let Some(parent) = stage_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            atomic_write(&stage_path, &read.bytes, Some(metadata.mode.max(0o400)))?;
        }
        let files = collect_files(&baseline_root)?;
        let staged = collect_files(&stage_root)?;
        if files != staged {
            bail!("the staged snapshot does not match its baseline; refusing to continue");
        }
        let state = WorkspaceState {
            project_root: project_root.clone(),
            stage_root: stage_root.clone(),
            baseline_root: baseline_root.clone(),
            files,
        };
        let workspace = Self { state, warnings };
        workspace.save(&run_dir)?;
        Ok(workspace)
    }

    pub fn open(run_dir: &Path) -> Result<Self> {
        let run_dir = run_dir
            .canonicalize()
            .with_context(|| format!("failed to resolve run directory {}", run_dir.display()))?;
        let path = run_dir.join("workspace.json");
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("failed to stat workspace state {}", path.display()))?;
        if !metadata.is_file() {
            bail!("workspace state is not a regular file");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o077 != 0 {
                bail!("workspace state must not be readable by other users");
            }
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read workspace state {}", path.display()))?;
        let state: WorkspaceState = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse workspace state {}", path.display()))?;
        let expected_stage = run_dir.join("workspace");
        let expected_baseline = run_dir.join("baseline");
        if state.stage_root != expected_stage || state.baseline_root != expected_baseline {
            bail!("workspace state points outside its run directory");
        }
        if !state.stage_root.is_dir() || !state.baseline_root.is_dir() {
            bail!("workspace snapshot directories are missing");
        }
        let baseline = collect_files(&state.baseline_root)?;
        if baseline != state.files {
            bail!("baseline snapshot does not match the recorded manifest");
        }
        Ok(Self {
            state,
            warnings: Vec::new(),
        })
    }

    pub fn save(&self, run_dir: &Path) -> Result<()> {
        let data = serde_json::to_vec_pretty(&self.state)?;
        atomic_write(&run_dir.join("workspace.json"), &data, Some(0o600))
    }

    pub fn project_root(&self) -> &Path {
        &self.state.project_root
    }

    pub fn stage_root(&self) -> &Path {
        &self.state.stage_root
    }

    pub fn baseline_root(&self) -> &Path {
        &self.state.baseline_root
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn baseline_files(&self) -> &BTreeMap<String, FileEntry> {
        &self.state.files
    }

    pub fn current_files(&self) -> Result<BTreeMap<String, FileEntry>> {
        collect_files(&self.state.stage_root)
    }

    pub fn tree_hash(&self) -> Result<String> {
        tree_hash(&self.current_files()?)
    }

    pub fn diff(&self, max_bytes: usize) -> Result<DiffReport> {
        compute_diff(&self.state.stage_root, &self.state.baseline_root, max_bytes)
    }

    /// Apply exactly the reviewed bytes. `authorization` must name the current
    /// staged snapshot and the evaluation that approved it; evidence for any
    /// other tree is rejected.
    pub fn apply(
        &self,
        run_dir: &Path,
        authorization: &ApplyAuthorization,
        evidence: &ApplyEvidence,
    ) -> Result<ApplyReport> {
        let stage_files = self.current_files()?;
        let snapshot_hash = tree_hash(&stage_files)?;
        if authorization.snapshot_hash != snapshot_hash {
            bail!(
                "the staged snapshot changed after review; re-evaluate and approve the current diff"
            );
        }
        if evidence.snapshot_hash != snapshot_hash {
            bail!("evaluation evidence does not belong to the current staged snapshot");
        }
        if evidence.id != authorization.evaluation_id {
            bail!("apply approval does not match the evaluation evidence");
        }
        if !evidence.passed {
            bail!("refusing to apply: the evaluator did not pass for this snapshot");
        }
        let operations = changed_files(&self.state.files, &stage_files);
        if operations.is_empty() {
            bail!("there are no staged changes to apply");
        }

        let project_guard =
            ToolContext::new(&self.state.project_root, None, 1024 * 1024, 1024 * 1024)?;
        let mut conflicts = Vec::new();
        for operation in &operations {
            match operation.status {
                ChangeStatus::Added => {
                    if project_guard.exists(&operation.path) {
                        conflicts.push(format!("{} now exists in the project", operation.path));
                    }
                }
                ChangeStatus::Modified | ChangeStatus::Deleted => {
                    let expected = operation
                        .old_hash
                        .as_deref()
                        .context("internal error: missing old hash")?;
                    match project_guard.read_bounded(&operation.path, 1) {
                        Ok(read) => {
                            if read.hash != expected {
                                conflicts.push(format!(
                                    "{} changed in the project after planning",
                                    operation.path
                                ));
                                continue;
                            }
                            let expected_mode = self
                                .state
                                .files
                                .get(&operation.path)
                                .and_then(|entry| entry.mode);
                            if expected_mode.is_some()
                                && project_guard.mode_of(&operation.path) != expected_mode
                            {
                                conflicts.push(format!(
                                    "{} changed permissions in the project after planning",
                                    operation.path
                                ));
                            }
                        }
                        Err(_) => conflicts.push(format!(
                            "{} was removed from the project after planning",
                            operation.path
                        )),
                    }
                }
            }
        }
        if !conflicts.is_empty() {
            bail!(
                "refusing to apply because the project changed:\n{}",
                conflicts.join("\n")
            );
        }

        let apply_id = uuid::Uuid::new_v4().simple().to_string();
        let backup_dir = run_dir.join("backups").join(format!("apply-{apply_id}"));
        create_private_dir(&backup_dir)?;
        for operation in &operations {
            if matches!(
                operation.status,
                ChangeStatus::Modified | ChangeStatus::Deleted
            ) {
                let read = project_guard
                    .read_bounded(&operation.path, MAX_APPLY_FILE_BYTES)
                    .with_context(|| format!("failed to back up {}", operation.path))?;
                if read.truncated {
                    bail!(
                        "{} is too large to back up safely (limit {} bytes)",
                        operation.path,
                        crate::util::display_bytes(MAX_APPLY_FILE_BYTES)
                    );
                }
                let backup = backup_dir.join(&operation.path);
                if let Some(parent) = backup.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mode = project_guard.mode_of(&operation.path).or(Some(0o600));
                atomic_write(&backup, &read.bytes, mode)?;
            }
        }

        let journal_path = run_dir.join("apply-journal.json");
        let mut journal = ApplyJournal {
            id: apply_id,
            project_root: self.state.project_root.clone(),
            backup_dir: backup_dir.clone(),
            evaluation_id: evidence.id.clone(),
            snapshot_hash: snapshot_hash.clone(),
            state: "prepared".to_owned(),
            operations: operations
                .iter()
                .map(|file| ApplyOperation {
                    file: file.clone(),
                    state: "pending".to_owned(),
                    error: None,
                })
                .collect(),
            rollback_errors: Vec::new(),
            error: None,
        };
        write_journal(&journal_path, &journal)?;
        let stage_handle = DirHandle::open_root(&self.state.stage_root)?;

        for (index, operation) in operations.iter().enumerate() {
            let result = apply_one(&project_guard, &stage_handle, &stage_files, operation);
            match result {
                Ok(()) => {
                    journal.operations[index].state = "applied".to_owned();
                    write_journal(&journal_path, &journal)?;
                }
                Err(error) => {
                    let message = format!("{error:#}");
                    journal.operations[index].state = "failed".to_owned();
                    journal.operations[index].error = Some(message.clone());
                    let applied: Vec<&ChangedFile> = operations[..index].iter().collect();
                    let errors = rollback(&project_guard, &backup_dir, &applied, &self.state.files);
                    for (rollback_operation, error) in errors {
                        if let Some(entry) = journal
                            .operations
                            .iter_mut()
                            .find(|entry| entry.file.path == rollback_operation)
                        {
                            entry.state = "rollback_failed".to_owned();
                            entry.error = Some(error.clone());
                        }
                        journal
                            .rollback_errors
                            .push(format!("{rollback_operation}: {error}"));
                    }
                    journal.error = Some(message.clone());
                    journal.state = if journal.rollback_errors.is_empty() {
                        "rolled_back".to_owned()
                    } else {
                        "rollback_failed".to_owned()
                    };
                    write_journal(&journal_path, &journal)?;
                    if journal.rollback_errors.is_empty() {
                        bail!("apply failed and was rolled back: {message}");
                    }
                    bail!(
                        "apply failed and rollback was incomplete: {message}\nrollback failures:\n{}",
                        journal.rollback_errors.join("\n")
                    );
                }
            }
        }
        journal.state = "applied".to_owned();
        write_journal(&journal_path, &journal)?;
        Ok(ApplyReport {
            applied: operations,
            backup_dir,
            journal: journal_path,
        })
    }
}

fn changed_files(
    baseline: &BTreeMap<String, FileEntry>,
    stage: &BTreeMap<String, FileEntry>,
) -> Vec<ChangedFile> {
    let paths: BTreeSet<&String> = baseline.keys().chain(stage.keys()).collect();
    let mut changes = Vec::new();
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
        changes.push(ChangedFile {
            path: path.clone(),
            status,
            old_hash: before.map(|entry| entry.hash.clone()),
            new_hash: after.map(|entry| entry.hash.clone()),
        });
    }
    changes
}

fn apply_one(
    project_guard: &ToolContext,
    stage_handle: &DirHandle,
    stage_files: &BTreeMap<String, FileEntry>,
    operation: &ChangedFile,
) -> Result<()> {
    match operation.status {
        ChangeStatus::Added => {
            if project_guard.exists(&operation.path) {
                bail!("{} appeared during apply", operation.path);
            }
            let (data, entry) =
                read_verified_stage_file(stage_handle, stage_files, &operation.path)?;
            project_guard.write_file(&operation.path, &data, entry.mode)
        }
        ChangeStatus::Modified => {
            let expected = operation
                .old_hash
                .as_deref()
                .context("internal error: missing old hash")?;
            let actual = project_guard
                .read_bounded(&operation.path, 1)
                .with_context(|| format!("failed to verify {}", operation.path))?
                .hash;
            if actual != expected {
                bail!("{} changed during apply", operation.path);
            }
            let (data, entry) =
                read_verified_stage_file(stage_handle, stage_files, &operation.path)?;
            project_guard.write_file(&operation.path, &data, entry.mode)
        }
        ChangeStatus::Deleted => {
            let expected = operation
                .old_hash
                .as_deref()
                .context("internal error: missing old hash")?;
            let actual = project_guard
                .read_bounded(&operation.path, 1)
                .with_context(|| format!("failed to verify {}", operation.path))?
                .hash;
            if actual != expected {
                bail!("{} changed during apply", operation.path);
            }
            project_guard.remove_file(&operation.path)
        }
    }
}

fn read_verified_stage_file<'a>(
    stage_handle: &DirHandle,
    stage_files: &'a BTreeMap<String, FileEntry>,
    relative: &str,
) -> Result<(Vec<u8>, &'a FileEntry)> {
    let read = stage_handle
        .read(relative, MAX_APPLY_FILE_BYTES)
        .with_context(|| format!("failed to read staged {relative}"))?;
    if read.truncated {
        bail!("staged file {relative} is too large to apply safely");
    }
    let entry = stage_files
        .get(relative)
        .context("internal error: staged file missing from manifest")?;
    let actual = hash_bytes(&read.bytes);
    if actual != entry.hash {
        bail!("staged file {relative} changed after review; refusing to apply");
    }
    Ok((read.bytes, entry))
}

/// Restore already-applied operations. Returns one `(path, error)` pair per
/// failed restore so the journal can report partial recovery truthfully.
fn rollback(
    project_guard: &ToolContext,
    backup_dir: &Path,
    applied: &[&ChangedFile],
    baseline: &BTreeMap<String, FileEntry>,
) -> Vec<(String, String)> {
    let mut failures = Vec::new();
    for operation in applied.iter().rev() {
        if baseline.contains_key(&operation.path) {
            let backup = backup_dir.join(&operation.path);
            match std::fs::read(&backup) {
                Ok(data) => {
                    let mode = baseline
                        .get(&operation.path)
                        .and_then(|entry| entry.mode)
                        .or_else(|| file_mode(&backup));
                    if let Err(error) = project_guard.write_file(&operation.path, &data, mode) {
                        failures.push((operation.path.clone(), format!("{error:#}")));
                    }
                }
                Err(error) => failures.push((
                    operation.path.clone(),
                    format!("failed to read backup {}: {error}", backup.display()),
                )),
            }
        } else if project_guard.exists(&operation.path)
            && let Err(error) = project_guard.remove_file(&operation.path)
        {
            failures.push((operation.path.clone(), format!("{error:#}")));
        }
    }
    failures
}

fn write_journal(path: &Path, journal: &ApplyJournal) -> Result<()> {
    let data = serde_json::to_vec_pretty(journal)?;
    atomic_write(path, &data, Some(0o600))
}

fn create_private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn source_paths(root: &Path) -> Result<(Vec<String>, Vec<String>)> {
    if let Some(paths) = git_paths(root)? {
        let mut filtered = Vec::new();
        let mut warnings = Vec::new();
        for path in paths {
            let relative = Path::new(&path);
            // Tracked-but-ignored files are preserved: only secrets are
            // removed, never files merely because an ignore rule matches.
            if is_sensitive_rel(relative) {
                continue;
            }
            filtered.push(path);
        }
        if filtered.len() > MAX_FILES {
            bail!("repository exceeds the {MAX_FILES} file snapshot limit");
        }
        return Ok((filtered, std::mem::take(&mut warnings)));
    }

    let mut paths = Vec::new();
    let mut warnings = Vec::new();
    for entry in walker(root)? {
        let entry = entry.with_context(|| {
            format!(
                "failed to enumerate {}; refusing to create an incomplete baseline",
                root.display()
            )
        })?;
        if entry
            .file_type()
            .is_some_and(|file_type| file_type.is_symlink())
        {
            if let Some(relative) = relative_string(root, entry.path()) {
                warnings.push(format!("skipped symlink {relative}"));
                if warnings.len() > MAX_WARNINGS {
                    bail!("too many unreadable entries; refusing an incomplete baseline");
                }
            }
            continue;
        }
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }
        if let Some(relative) = relative_string(root, entry.path()) {
            paths.push(relative);
        }
        if paths.len() > MAX_FILES {
            bail!("repository exceeds the {MAX_FILES} file snapshot limit");
        }
    }
    Ok((paths, warnings))
}

/// Run git with a cleared environment so repository configuration cannot
/// select external helpers, credential prompts or hooks while we only read.
fn git_command(root: &Path) -> Command {
    let mut command = Command::new("git");
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_OPTIONAL_LOCKS", "0")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("HOME", "/nonexistent")
        .arg("-c")
        .arg("core.fsmonitor=false")
        .arg("-c")
        .arg("core.untrackedCache=false")
        .arg("-C")
        .arg(root);
    command
}

fn git_paths(root: &Path) -> Result<Option<Vec<String>>> {
    let top = git_command(root)
        .args(["rev-parse", "--show-toplevel"])
        .output();
    let Ok(top) = top else {
        return Ok(None);
    };
    if !top.status.success() {
        return Ok(None);
    }
    let top = String::from_utf8_lossy(&top.stdout).trim().to_owned();
    let Ok(top) = Path::new(&top).canonicalize() else {
        return Ok(None);
    };
    if top != root {
        return Ok(None);
    }
    let output = git_command(root)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .output()
        .context("failed to invoke git ls-files")?;
    if !output.status.success() {
        return Ok(None);
    }
    let mut paths = Vec::new();
    for entry in output.stdout.split(|byte| *byte == 0) {
        if entry.is_empty() {
            continue;
        }
        match std::str::from_utf8(entry) {
            Ok(path) => paths.push(path.to_owned()),
            Err(_) => bail!("git returned a non-UTF-8 path; refusing an incomplete baseline"),
        }
    }
    paths.sort();
    paths.dedup();
    Ok(Some(paths))
}

pub fn tree_hash(files: &BTreeMap<String, FileEntry>) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    for (path, entry) in files {
        hasher.update(path.as_bytes());
        hasher.update(&[0]);
        hasher.update(entry.hash.as_bytes());
        hasher.update(&[0]);
        hasher.update(entry.mode.unwrap_or(0).to_le_bytes().as_slice());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn hash_text(bytes: &[u8]) -> String {
    hash_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn init_repo() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        Command::new("git")
            .arg("-C")
            .arg(directory.path())
            .args(["init", "-q"])
            .status()
            .unwrap();
        Command::new("git")
            .arg("-C")
            .arg(directory.path())
            .args(["config", "user.email", "test@example.com"])
            .status()
            .unwrap();
        Command::new("git")
            .arg("-C")
            .arg(directory.path())
            .args(["config", "user.name", "Test"])
            .status()
            .unwrap();
        std::fs::write(directory.path().join("tracked.txt"), "tracked\n").unwrap();
        std::fs::write(directory.path().join(".env"), "SECRET=1\n").unwrap();
        std::fs::write(directory.path().join("untracked.txt"), "untracked\n").unwrap();
        Command::new("git")
            .arg("-C")
            .arg(directory.path())
            .args(["add", "tracked.txt"])
            .status()
            .unwrap();
        Command::new("git")
            .arg("-C")
            .arg(directory.path())
            .args(["commit", "-qm", "initial"])
            .status()
            .unwrap();
        directory
    }

    fn authorization(workspace: &Workspace) -> (ApplyAuthorization, ApplyEvidence) {
        let snapshot_hash = workspace.tree_hash().unwrap();
        let evaluation_id = "eval-test".to_owned();
        (
            ApplyAuthorization {
                snapshot_hash: snapshot_hash.clone(),
                evaluation_id: evaluation_id.clone(),
                approved_at_ms: 0,
            },
            ApplyEvidence {
                id: evaluation_id,
                snapshot_hash,
                policy_hash: "policy".to_owned(),
                passed: true,
            },
        )
    }

    #[test]
    fn snapshots_tracked_untracked_and_skips_secrets() {
        let project = init_repo();
        let run = tempfile::tempdir().unwrap();
        let workspace = Workspace::create(project.path(), run.path()).unwrap();
        assert!(workspace.stage_root().join("tracked.txt").exists());
        assert!(workspace.stage_root().join("untracked.txt").exists());
        assert!(!workspace.stage_root().join(".env").exists());
    }

    #[test]
    fn applies_changes_and_preserves_untouched_files() {
        let project = init_repo();
        let run = tempfile::tempdir().unwrap();
        let workspace = Workspace::create(project.path(), run.path()).unwrap();
        std::fs::write(workspace.stage_root().join("tracked.txt"), "changed\n").unwrap();
        std::fs::write(workspace.stage_root().join("new.txt"), "new\n").unwrap();
        std::fs::remove_file(workspace.stage_root().join("untracked.txt")).unwrap();
        let (authorization, evidence) = authorization(&workspace);
        let report = workspace
            .apply(run.path(), &authorization, &evidence)
            .unwrap();
        assert_eq!(report.applied.len(), 3);
        assert_eq!(
            std::fs::read_to_string(project.path().join("tracked.txt")).unwrap(),
            "changed\n"
        );
        assert_eq!(
            std::fs::read_to_string(project.path().join("new.txt")).unwrap(),
            "new\n"
        );
        assert!(!project.path().join("untracked.txt").exists());
        assert_eq!(
            std::fs::read_to_string(project.path().join(".env")).unwrap(),
            "SECRET=1\n"
        );
    }

    #[test]
    fn refuses_to_overwrite_a_changed_project() {
        let project = init_repo();
        let run = tempfile::tempdir().unwrap();
        let workspace = Workspace::create(project.path(), run.path()).unwrap();
        std::fs::write(workspace.stage_root().join("tracked.txt"), "staged\n").unwrap();
        std::fs::write(project.path().join("tracked.txt"), "user edit\n").unwrap();
        let (authorization, evidence) = authorization(&workspace);
        let error = workspace
            .apply(run.path(), &authorization, &evidence)
            .unwrap_err();
        assert!(format!("{error:#}").contains("changed in the project"));
        assert_eq!(
            std::fs::read_to_string(project.path().join("tracked.txt")).unwrap(),
            "user edit\n"
        );
    }

    #[test]
    fn refuses_new_file_that_appeared() {
        let project = init_repo();
        let run = tempfile::tempdir().unwrap();
        let workspace = Workspace::create(project.path(), run.path()).unwrap();
        std::fs::write(workspace.stage_root().join("new.txt"), "staged\n").unwrap();
        std::fs::write(project.path().join("new.txt"), "user\n").unwrap();
        let (authorization, evidence) = authorization(&workspace);
        assert!(
            workspace
                .apply(run.path(), &authorization, &evidence)
                .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(project.path().join("new.txt")).unwrap(),
            "user\n"
        );
    }

    #[test]
    fn refuses_a_run_directory_inside_the_project() {
        let project = init_repo();
        let inside = project.path().join("run");
        assert!(Workspace::create(project.path(), &inside).is_err());
    }

    #[test]
    fn skips_symlinks() {
        let project = init_repo();
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc/passwd", project.path().join("password-link")).unwrap();
        let run = tempfile::tempdir().unwrap();
        let workspace = Workspace::create(project.path(), run.path()).unwrap();
        assert!(!workspace.stage_root().join("password-link").exists());
    }
}
