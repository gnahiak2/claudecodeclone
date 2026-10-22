//! Private, versioned session state with exclusive locking and redacted,
//! append-only events. Sessions live outside the project tree.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::providers::Usage;
use crate::roles::planner::Plan;
use crate::util::{atomic_write, now_ms, redact};

pub const SESSION_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Planner,
    Executor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Created,
    Planning,
    PlanReview,
    Staging,
    Executing,
    Evaluating,
    Repairing,
    Review,
    ApplyReview,
    Applied,
    Completed,
    NeedsAttention,
    Failed,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Planning => "planning",
            Self::PlanReview => "plan_review",
            Self::Staging => "staging",
            Self::Executing => "executing",
            Self::Evaluating => "evaluating",
            Self::Repairing => "repairing",
            Self::Review => "review",
            Self::ApplyReview => "apply_review",
            Self::Applied => "applied",
            Self::Completed => "completed",
            Self::NeedsAttention => "needs_attention",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetState {
    pub planner_calls: u32,
    pub executor_calls: u32,
    pub tool_calls: u32,
    pub reserved_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub repairs: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRecord {
    pub kind: String,
    pub hash: String,
    pub approved_at_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub id: String,
    pub snapshot_hash: String,
    pub policy_hash: String,
    pub passed: bool,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplySummary {
    pub snapshot_hash: String,
    pub evaluation_id: String,
    pub state: String,
    pub journal: PathBuf,
    pub rollback_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub version: u32,
    pub id: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub task: String,
    pub project_root: PathBuf,
    pub run_dir: PathBuf,
    pub phase: Phase,
    pub plan: Option<Plan>,
    pub plan_hash: Option<String>,
    pub approvals: Vec<ApprovalRecord>,
    pub budget: BudgetState,
    pub evaluation: Option<EvaluationSummary>,
    pub apply: Option<ApplySummary>,
    pub error: Option<String>,
}

impl SessionState {
    pub fn approval(&self, kind: &str, hash: &str) -> bool {
        self.approvals
            .iter()
            .any(|record| record.kind == kind && record.hash == hash)
    }

    pub fn record_approval(&mut self, kind: &str, hash: &str) {
        self.approvals.push(ApprovalRecord {
            kind: kind.to_owned(),
            hash: hash.to_owned(),
            approved_at_ms: now_ms(),
        });
    }
}

#[derive(Debug)]
pub struct SessionStore {
    directory: PathBuf,
    _lock: SessionLock,
}

#[derive(Debug)]
struct SessionLock {
    path: PathBuf,
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl SessionStore {
    pub fn create(
        sessions_root: &Path,
        project_root: &Path,
        task: &str,
    ) -> Result<(Self, SessionState)> {
        let project_root = project_root
            .canonicalize()
            .with_context(|| format!("failed to resolve project {}", project_root.display()))?;
        let root = sessions_root.to_path_buf();
        if root.starts_with(&project_root) {
            bail!("sessions_dir must be outside the project");
        }
        create_private_dir(&root)?;
        let root = root
            .canonicalize()
            .with_context(|| "failed to resolve sessions directory")?;
        let id = format!("ses_{}", uuid::Uuid::new_v4().simple());
        let directory = root.join(&id);
        create_private_dir(&directory)?;
        let lock = SessionLock::acquire(&directory)?;
        let run_dir = directory.join("run");
        if run_dir.starts_with(&project_root) {
            bail!("run directory must be outside the project");
        }
        std::fs::create_dir_all(&run_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&run_dir, std::fs::Permissions::from_mode(0o700))?;
        }
        let now = now_ms();
        let mut state = SessionState {
            version: SESSION_FORMAT_VERSION,
            id: id.clone(),
            created_at_ms: now,
            updated_at_ms: now,
            task: task.to_owned(),
            project_root,
            run_dir,
            phase: Phase::Created,
            plan: None,
            plan_hash: None,
            approvals: Vec::new(),
            budget: BudgetState::default(),
            evaluation: None,
            apply: None,
            error: None,
        };
        let store = Self {
            directory,
            _lock: lock,
        };
        store.save(&mut state)?;
        Ok((store, state))
    }

    pub fn open(sessions_root: &Path, id: &str) -> Result<(Self, SessionState)> {
        if !id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            bail!("invalid session id");
        }
        let root = sessions_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve sessions directory {}",
                sessions_root.display()
            )
        })?;
        let directory = root.join(id);
        let canonical = directory
            .canonicalize()
            .with_context(|| format!("session {id} does not exist"))?;
        if !canonical.starts_with(&root) {
            bail!("session path escapes the sessions directory");
        }
        let lock = SessionLock::acquire(&canonical)?;
        let store = Self {
            directory: canonical,
            _lock: lock,
        };
        let state = store.load()?;
        Ok((store, state))
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn state_path(&self) -> PathBuf {
        self.directory.join("session.json")
    }

    pub fn id(&self) -> &str {
        self.directory
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("session")
    }

    pub fn load(&self) -> Result<SessionState> {
        let path = self.directory.join("session.json");
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("failed to stat {}", path.display()))?;
        if !metadata.is_file() {
            bail!("session state is not a regular file");
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if metadata.permissions().mode() & 0o077 != 0 {
                bail!("session state must not be accessible by other users");
            }
        }
        let text = std::fs::read_to_string(&path)?;
        let state: SessionState = serde_json::from_str(&text)
            .with_context(|| "session state is corrupt or from an unsupported version")?;
        if state.version != SESSION_FORMAT_VERSION {
            bail!(
                "unsupported session format version {} (expected {SESSION_FORMAT_VERSION})",
                state.version
            );
        }
        if state.id != self.id() {
            bail!("session state id does not match its directory");
        }
        if !state.run_dir.starts_with(&self.directory) {
            bail!("session run directory is outside the session");
        }
        if !state.project_root.is_absolute() {
            bail!("session project root is not absolute");
        }
        Ok(state)
    }

    pub fn save(&self, state: &mut SessionState) -> Result<()> {
        state.updated_at_ms = now_ms();
        let data = serde_json::to_vec_pretty(state)?;
        atomic_write(&self.directory.join("session.json"), &data, Some(0o600))
    }

    pub fn event(&self, kind: &str, details: serde_json::Value, secrets: &[String]) -> Result<()> {
        let record = serde_json::json!({
            "at_ms": now_ms(),
            "kind": kind,
            "details": details,
        });
        let mut line = serde_json::to_string(&record)?;
        line = redact(&line, secrets);
        let path = self.directory.join("events.jsonl");
        let mut options = OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Reconcile interrupted work without replaying mutations. Apply journals
    /// are reported, never resumed automatically.
    pub fn reconcile(&self, state: &mut SessionState) -> Result<Vec<String>> {
        let mut notes = Vec::new();
        let journal_path = state.run_dir.join("apply-journal.json");
        if journal_path.exists() {
            let text = std::fs::read_to_string(&journal_path)?;
            match serde_json::from_str::<crate::workspace::ApplyJournal>(&text) {
                Ok(journal) => {
                    let message = match journal.state.as_str() {
                        "applied" => None,
                        "prepared" => Some(
                            "an apply was interrupted while prepared; inspect the journal and project before retrying"
                                .to_owned(),
                        ),
                        "rolled_back" => Some(
                            "a previous apply failed and was rolled back; re-evaluate before applying again"
                                .to_owned(),
                        ),
                        _ => Some(format!(
                            "a previous apply ended in state {:?}; inspect {}",
                            journal.state,
                            journal_path.display()
                        )),
                    };
                    if let Some(message) = message {
                        notes.push(message);
                        state.phase = Phase::NeedsAttention;
                        state.apply = Some(ApplySummary {
                            snapshot_hash: journal.snapshot_hash,
                            evaluation_id: journal.evaluation_id,
                            state: journal.state,
                            journal: journal_path,
                            rollback_errors: journal.rollback_errors,
                        });
                    }
                }
                Err(error) => {
                    notes.push(format!("apply journal is unreadable: {error}"));
                    state.phase = Phase::NeedsAttention;
                }
            }
        }
        if state.phase == Phase::Executing
            || state.phase == Phase::Evaluating
            || state.phase == Phase::Repairing
        {
            notes.push(
                "the previous run was interrupted mid-flight; the staged workspace is preserved but no mutation was replayed"
                    .to_owned(),
            );
            state.phase = Phase::NeedsAttention;
        }
        Ok(notes)
    }
}

impl SessionLock {
    fn acquire(directory: &Path) -> Result<Self> {
        let path = directory.join("lock");
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .with_context(|| {
                format!(
                    "session is already in use (lock {} exists); remove it only if no relay process is running",
                    path.display()
                )
            })?;
        Ok(Self { path })
    }
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

/// Budget accounting shared with the role loops. Every mutation is persisted
/// before the associated request is sent.
pub trait BudgetSink {
    fn reserve_model(&self, role: Role, max_tokens: u32) -> Result<()>;
    fn reserve_tool(&self) -> Result<()>;
    fn record_usage(&self, usage: &Usage) -> Result<()>;
}

pub struct BudgetLimits {
    pub planner_calls: u32,
    pub executor_calls: u32,
    pub tool_calls: u32,
    pub tokens: u64,
    pub context_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_lock_prevents_concurrent_opens() {
        let root = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let (_store, state) = SessionStore::create(root.path(), project.path(), "task").unwrap();
        let id = state.id.clone();
        assert!(SessionStore::open(root.path(), &id).is_err());
    }

    #[test]
    fn rejects_session_ids_with_traversal() {
        let root = tempfile::tempdir().unwrap();
        assert!(SessionStore::open(root.path(), "../escape").is_err());
        assert!(SessionStore::open(root.path(), "a/b").is_err());
    }

    #[test]
    fn interrupted_apply_requires_attention_without_replay() {
        let root = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let (store, mut state) = SessionStore::create(root.path(), project.path(), "task").unwrap();
        let journal = serde_json::json!({
            "id": "apply-1",
            "project_root": project.path(),
            "backup_dir": state.run_dir.join("backups"),
            "evaluation_id": "eval_1",
            "snapshot_hash": "snap",
            "state": "prepared",
            "operations": [],
            "rollback_errors": [],
            "error": null
        });
        std::fs::write(
            state.run_dir.join("apply-journal.json"),
            serde_json::to_vec(&journal).unwrap(),
        )
        .unwrap();
        let notes = store.reconcile(&mut state).unwrap();
        assert_eq!(state.phase, Phase::NeedsAttention);
        assert!(notes.iter().any(|note| note.contains("interrupted")));
        assert_eq!(state.apply.as_ref().unwrap().state, "prepared");
    }

    #[test]
    fn events_are_redacted() {
        let root = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        let (store, _state) = SessionStore::create(root.path(), project.path(), "task").unwrap();
        store
            .event(
                "test",
                serde_json::json!({"message": "secret-value"}),
                &["secret-value".to_owned()],
            )
            .unwrap();
        let events = store.directory().join("events.jsonl");
        let text = std::fs::read_to_string(&events).unwrap();
        assert!(!text.contains("secret-value"));
        assert!(text.contains("[REDACTED]"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&events).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}
