mod docker;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::TestProfile;

use crate::util::{hash_bytes, truncate_bytes};
use crate::workspace::Workspace;

pub use docker::DockerEvaluator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandResult {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration_ms: u64,
    pub stdout: String,
    pub stderr: String,
    pub stdout_truncated: bool,
    pub stderr_truncated: bool,
    pub cleanup_error: Option<String>,
}

impl CommandResult {
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

/// One test profile as explicitly approved. Evidence binds to these exact
/// values, not to whatever is in the configuration file later.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovedProfile {
    pub name: String,
    pub image: String,
    pub command: Vec<String>,
    pub cwd: String,
    pub timeout_secs: u64,
    pub allow_network: bool,
    pub prepare_command: Vec<String>,
    pub prepare_allow_network: bool,
}

impl ApprovedProfile {
    pub fn from_test(test: &TestProfile) -> Self {
        Self {
            name: test.name.clone(),
            image: test.image.clone(),
            command: test.command.clone(),
            cwd: test.cwd.clone(),
            timeout_secs: test.timeout_secs,
            allow_network: test.allow_network,
            prepare_command: test.prepare_command.clone(),
            prepare_allow_network: test.prepare_allow_network,
        }
    }
}

/// The complete approved evaluation policy. Changing any byte changes the
/// policy hash and invalidates prior evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationPolicy {
    pub profiles: Vec<ApprovedProfile>,
}

impl EvaluationPolicy {
    pub fn from_tests(tests: &[TestProfile]) -> Self {
        Self {
            profiles: tests.iter().map(ApprovedProfile::from_test).collect(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.profiles.is_empty() {
            bail!("evaluation requires at least one approved test profile");
        }
        let mut names = BTreeSet::new();
        for profile in &self.profiles {
            if !names.insert(&profile.name) {
                bail!("duplicate approved test profile {:?}", profile.name);
            }
            // Reuse the configuration validator so an approved policy cannot
            // smuggle in an invalid image, argv, cwd or timeout.
            TestProfile {
                name: profile.name.clone(),
                image: profile.image.clone(),
                command: profile.command.clone(),
                cwd: profile.cwd.clone(),
                timeout_secs: profile.timeout_secs,
                allow_network: profile.allow_network,
                prepare_command: profile.prepare_command.clone(),
                prepare_allow_network: profile.prepare_allow_network,
            }
            .validate()?;
        }
        Ok(())
    }

    pub fn hash(&self) -> String {
        let encoded = serde_json::to_vec(self).unwrap_or_default();
        hash_bytes(&encoded)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestOutcome {
    pub name: String,
    pub image: String,
    pub image_id: Option<String>,
    pub command: Vec<String>,
    pub cwd: String,
    pub allow_network: bool,
    pub prepare_allow_network: bool,
    pub snapshot_hash: String,
    pub policy_hash: String,
    pub passed: bool,
    pub setup_failure: Option<String>,
    pub prepare: Option<CommandResult>,
    pub result: Option<CommandResult>,
    pub cleanup_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub id: String,
    pub snapshot_hash: String,
    pub policy_hash: String,
    pub outcomes: Vec<TestOutcome>,
    pub all_passed: bool,
}

impl EvaluationReport {
    pub fn evidence(&self) -> crate::workspace::ApplyEvidence {
        crate::workspace::ApplyEvidence {
            id: self.id.clone(),
            snapshot_hash: self.snapshot_hash.clone(),
            policy_hash: self.policy_hash.clone(),
            passed: self.all_passed,
        }
    }
}

#[async_trait]
pub trait Evaluator: Send + Sync {
    async fn preflight(&self, policy: &EvaluationPolicy) -> Result<BTreeMap<String, String>>;

    async fn evaluate(
        &self,
        workspace: &Workspace,
        policy: &EvaluationPolicy,
        run_dir: &Path,
        run_id: &str,
    ) -> Result<EvaluationReport>;
}

pub fn all_passed(outcomes: &[TestOutcome]) -> bool {
    !outcomes.is_empty()
        && outcomes.iter().all(|outcome| {
            outcome.passed && outcome.setup_failure.is_none() && outcome.cleanup_error.is_none()
        })
}

pub fn failure_evidence(outcomes: &[TestOutcome], max_bytes: usize) -> String {
    let mut output = String::new();
    for outcome in outcomes.iter().filter(|outcome| !outcome.passed) {
        output.push_str(&format!(
            "test profile: {}\nimage: {}\ncommand: {:?}\ncwd: {}\n",
            outcome.name, outcome.image, outcome.command, outcome.cwd
        ));
        if let Some(setup) = &outcome.setup_failure {
            output.push_str(&format!("setup failure: {setup}\n"));
        }
        if let Some(prepare) = &outcome.prepare
            && !prepare.succeeded()
        {
            output.push_str(&format!(
                "prepare failed (exit {:?}, timed out: {})\nstdout:\n{}\nstderr:\n{}\n",
                prepare.exit_code, prepare.timed_out, prepare.stdout, prepare.stderr
            ));
        }
        if let Some(result) = &outcome.result {
            output.push_str(&format!(
                "test failed (exit {:?}, timed out: {})\nstdout:\n{}\nstderr:\n{}\n",
                result.exit_code, result.timed_out, result.stdout, result.stderr
            ));
        }
        if let Some(error) = &outcome.cleanup_error {
            output.push_str(&format!("cleanup error: {error}\n"));
        }
        if output.len() >= max_bytes {
            break;
        }
    }
    crate::util::truncate_bytes(&output, max_bytes)
}

pub fn summary(outcomes: &[TestOutcome]) -> String {
    outcomes
        .iter()
        .map(|outcome| {
            let state = if outcome.passed {
                "passed"
            } else if outcome.setup_failure.is_some() {
                "setup failed"
            } else {
                "failed"
            };
            format!("{}: {}", outcome.name, state)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn display_result(result: &CommandResult) -> String {
    let mut output = String::new();
    if result.timed_out {
        output.push_str("timed out\n");
    }
    output.push_str(&format!("exit: {:?}\n", result.exit_code));
    if !result.stdout.is_empty() {
        output.push_str("stdout:\n");
        output.push_str(&truncate_bytes(&result.stdout, 8192));
        output.push('\n');
    }
    if !result.stderr.is_empty() {
        output.push_str("stderr:\n");
        output.push_str(&truncate_bytes(&result.stderr, 8192));
        output.push('\n');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> EvaluationPolicy {
        EvaluationPolicy {
            profiles: vec![ApprovedProfile {
                name: "unit".to_owned(),
                image: "rust:1.88".to_owned(),
                command: vec!["cargo".to_owned(), "test".to_owned()],
                cwd: ".".to_owned(),
                timeout_secs: 60,
                allow_network: false,
                prepare_command: vec![],
                prepare_allow_network: false,
            }],
        }
    }

    #[test]
    fn policy_hash_changes_with_command_image_and_network() {
        let base = policy();
        let mut changed = base.clone();
        changed.profiles[0].command.push("--all".to_owned());
        assert_ne!(base.hash(), changed.hash());

        let mut changed = base.clone();
        changed.profiles[0].image = "other:latest".to_owned();
        assert_ne!(base.hash(), changed.hash());

        let mut changed = base.clone();
        changed.profiles[0].allow_network = true;
        assert_ne!(base.hash(), changed.hash());
    }

    #[test]
    fn setup_failure_never_counts_as_pass() {
        let outcome = TestOutcome {
            name: "unit".to_owned(),
            image: "img".to_owned(),
            image_id: None,
            command: vec!["true".to_owned()],
            cwd: ".".to_owned(),
            allow_network: false,
            prepare_allow_network: false,
            snapshot_hash: "snap".to_owned(),
            policy_hash: "policy".to_owned(),
            passed: true,
            setup_failure: Some("image missing".to_owned()),
            prepare: None,
            result: None,
            cleanup_error: None,
        };
        assert!(!all_passed(std::slice::from_ref(&outcome)));
    }

    #[test]
    fn empty_profiles_are_rejected() {
        let policy = EvaluationPolicy { profiles: vec![] };
        assert!(policy.validate().is_err());
    }
}
