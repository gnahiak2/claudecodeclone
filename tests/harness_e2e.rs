use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use relay_code::config::{
    Config, DockerConfig, HarnessConfig, ProviderKind, ProviderSettings, ProvidersConfig,
    RoleConfig, RolesConfig, TestProfile,
};
use relay_code::evaluator::{
    CommandResult, EvaluationPolicy, EvaluationReport, Evaluator, TestOutcome,
};
use relay_code::harness::{
    ApprovalDecision, ApprovalRequest, Approver, Harness, HarnessBuilder, RejectAllApprover,
};
use relay_code::providers::scripted::{ScriptedProvider, text_response, tool_response};
use relay_code::providers::{CompletionRequest, CompletionResponse, ModelProvider};
use relay_code::util::hash_bytes;
use relay_code::workspace::{ApplyAuthorization, ApplyEvidence, Workspace, tree_hash};

struct RecordingApprover {
    decisions: VecDeque<ApprovalDecision>,
    kinds: Arc<Mutex<Vec<String>>>,
}

impl RecordingApprover {
    fn approve_all(count: usize) -> (Self, Arc<Mutex<Vec<String>>>) {
        let kinds = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                decisions: (0..count).map(|_| ApprovalDecision::Approved).collect(),
                kinds: kinds.clone(),
            },
            kinds,
        )
    }
}

impl Approver for RecordingApprover {
    fn approve(&mut self, request: ApprovalRequest) -> Result<ApprovalDecision> {
        let kind = match &request {
            ApprovalRequest::Plan { .. } => "plan",
            ApprovalRequest::Evaluation { .. } => "evaluation",
            ApprovalRequest::Apply { .. } => "apply",
        };
        self.kinds.lock().unwrap().push(kind.to_owned());
        Ok(self
            .decisions
            .pop_front()
            .unwrap_or_else(|| ApprovalDecision::Rejected("unexpected approval".to_owned())))
    }
}

struct FakeEvaluator {
    calls: Mutex<usize>,
}

#[async_trait]
impl Evaluator for FakeEvaluator {
    async fn preflight(
        &self,
        _policy: &EvaluationPolicy,
    ) -> Result<std::collections::BTreeMap<String, String>> {
        Ok(std::collections::BTreeMap::new())
    }

    async fn evaluate(
        &self,
        workspace: &Workspace,
        policy: &EvaluationPolicy,
        _run_dir: &Path,
        _run_id: &str,
    ) -> Result<EvaluationReport> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        let call = *calls;
        let content = std::fs::read_to_string(workspace.stage_root().join("greeting.txt"))
            .unwrap_or_default();
        let passed = content == "new\n";
        let snapshot = tree_hash(&workspace.current_files()?)?;
        let profile = &policy.profiles[0];
        let outcome = TestOutcome {
            name: profile.name.clone(),
            image: profile.image.clone(),
            image_id: Some("sha256:fake".to_owned()),
            command: profile.command.clone(),
            cwd: profile.cwd.clone(),
            allow_network: profile.allow_network,
            prepare_allow_network: profile.prepare_allow_network,
            snapshot_hash: snapshot.clone(),
            policy_hash: policy.hash(),
            passed,
            setup_failure: None,
            prepare: None,
            result: Some(CommandResult {
                exit_code: Some(if passed { 0 } else { 1 }),
                timed_out: false,
                duration_ms: 1,
                stdout: if passed {
                    "ok".to_owned()
                } else {
                    format!("call {call}: greeting is {:?}", content.trim())
                },
                stderr: String::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                cleanup_error: None,
            }),
            cleanup_error: None,
        };
        Ok(EvaluationReport {
            id: format!("eval_{call}"),
            snapshot_hash: snapshot,
            policy_hash: policy.hash(),
            all_passed: passed,
            outcomes: vec![outcome],
        })
    }
}

fn plan_json() -> String {
    serde_json::json!({
        "summary": "Replace the greeting",
        "steps": [{"title": "Edit", "details": "change greeting.txt"}],
        "files": [{"path": "greeting.txt", "change": "new value"}],
        "acceptance_criteria": ["greeting is new"],
        "risks": []
    })
    .to_string()
}

fn patch_response(expected: &str, new_text: &str) -> CompletionResponse {
    tool_response(
        "apply_patch",
        serde_json::json!({"edits": [{
            "path": "greeting.txt",
            "expected_hash": hash_bytes(expected.as_bytes()),
            "new_text": new_text
        }]}),
    )
}

fn config(sessions: &Path) -> Config {
    let config = Config {
        harness: HarnessConfig {
            sessions_dir: Some(sessions.to_path_buf()),
            max_repairs: 2,
            ..HarnessConfig::default()
        },
        docker: DockerConfig::default(),
        roles: RolesConfig {
            planner: RoleConfig {
                provider: ProviderKind::Ollama,
                model: "local".to_owned(),
                max_tokens: 1000,
                temperature: 0.1,
            },
            executor: RoleConfig {
                provider: ProviderKind::Ollama,
                model: "local".to_owned(),
                max_tokens: 1000,
                temperature: 0.1,
            },
        },
        providers: ProvidersConfig {
            ollama: Some(ProviderSettings {
                base_url: Some("http://localhost:11434".to_owned()),
                ..ProviderSettings::default()
            }),
            ..ProvidersConfig::default()
        },
        tests: vec![TestProfile {
            name: "fake".to_owned(),
            image: "fake:latest".to_owned(),
            command: vec!["true".to_owned()],
            cwd: ".".to_owned(),
            timeout_secs: 10,
            allow_network: false,
            prepare_command: vec![],
            prepare_allow_network: false,
        }],
    };
    config.validate().expect("valid config");
    config
}

#[tokio::test]
async fn plans_executes_fails_repairs_passes_reviews_and_applies() {
    let project = tempfile::tempdir().unwrap();
    let sessions = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("greeting.txt"), "old\n").unwrap();

    let planner = Arc::new(ScriptedProvider::new(vec![text_response(plan_json())]));
    let executor = Arc::new(ScriptedProvider::new(vec![
        patch_response("old\n", "broken\n"),
        text_response("first attempt"),
        patch_response("broken\n", "new\n"),
        text_response("repaired"),
    ]));
    let evaluator = Arc::new(FakeEvaluator {
        calls: Mutex::new(0),
    });
    let (approver, kinds) = RecordingApprover::approve_all(3);

    let harness = Harness::create(HarnessBuilder {
        config: config(sessions.path()),
        project_root: project.path().to_path_buf(),
        task: "replace the greeting".to_owned(),
        approver: Box::new(approver),
        resume_session: None,
        sessions_root: None,
        planner_override: Some(planner.clone()),
        executor_override: Some(executor.clone()),
        evaluator_override: Some(evaluator.clone()),
    })
    .unwrap();
    let session_id = harness.session_id().to_owned();
    let outcome = harness.run().await.unwrap();

    assert_eq!(
        std::fs::read_to_string(project.path().join("greeting.txt")).unwrap(),
        "new\n"
    );
    assert_eq!(outcome.phase.as_str(), "completed");
    assert_eq!(outcome.applied.as_ref().unwrap().applied.len(), 1);
    assert_eq!(*evaluator.calls.lock().unwrap(), 2);
    assert_eq!(
        kinds.lock().unwrap().as_slice(),
        &[
            "plan".to_owned(),
            "evaluation".to_owned(),
            "apply".to_owned()
        ]
    );
    assert!(outcome.diff.as_ref().unwrap().patch.contains("+new"));

    // The session must record the repair and the apply, with private perms.
    let state_path = sessions.path().join(&session_id).join("session.json");
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&state_path).unwrap()).unwrap();
    assert_eq!(state["phase"], "completed");
    assert_eq!(state["budget"]["repairs"], 1);
    assert!(state["budget"]["executor_calls"].as_u64().unwrap() >= 2);
    assert!(state["approvals"].as_array().unwrap().len() >= 3);

    let events =
        std::fs::read_to_string(sessions.path().join(&session_id).join("events.jsonl")).unwrap();
    assert!(events.contains("\"kind\":\"repair\""));
    assert!(events.contains("\"kind\":\"applied\""));
}

#[tokio::test]
async fn rejecting_the_plan_leaves_the_project_untouched() {
    let project = tempfile::tempdir().unwrap();
    let sessions = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("greeting.txt"), "old\n").unwrap();

    let planner = Arc::new(ScriptedProvider::new(vec![text_response(plan_json())]));
    let executor = Arc::new(ScriptedProvider::new(vec![]));
    let harness = Harness::create(HarnessBuilder {
        config: config(sessions.path()),
        project_root: project.path().to_path_buf(),
        task: "replace the greeting".to_owned(),
        approver: Box::new(RejectAllApprover),
        resume_session: None,
        sessions_root: None,
        planner_override: Some(planner),
        executor_override: Some(executor),
        evaluator_override: Some(Arc::new(FakeEvaluator {
            calls: Mutex::new(0),
        })),
    })
    .unwrap();
    let session_id = harness.session_id().to_owned();
    assert!(harness.run().await.is_err());

    assert_eq!(
        std::fs::read_to_string(project.path().join("greeting.txt")).unwrap(),
        "old\n"
    );
    let session = sessions.path().join(&session_id);
    assert!(!session.join("run/workspace").exists());
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(session.join("session.json")).unwrap())
            .unwrap();
    assert_eq!(state["phase"], "failed");
    assert!(state["error"].as_str().unwrap().contains("plan rejected"));
}

#[tokio::test]
async fn rejecting_apply_leaves_the_project_untouched() {
    let project = tempfile::tempdir().unwrap();
    let sessions = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("greeting.txt"), "old\n").unwrap();

    let planner = Arc::new(ScriptedProvider::new(vec![text_response(plan_json())]));
    let executor = Arc::new(ScriptedProvider::new(vec![
        patch_response("old\n", "new\n"),
        text_response("done"),
    ]));
    let evaluator = Arc::new(FakeEvaluator {
        calls: Mutex::new(0),
    });
    let (mut approver, _) = RecordingApprover::approve_all(3);
    approver.decisions.pop_back();
    approver
        .decisions
        .push_back(ApprovalDecision::Rejected("no apply".to_owned()));

    let harness = Harness::create(HarnessBuilder {
        config: config(sessions.path()),
        project_root: project.path().to_path_buf(),
        task: "replace the greeting".to_owned(),
        approver: Box::new(approver),
        resume_session: None,
        sessions_root: None,
        planner_override: Some(planner),
        executor_override: Some(executor),
        evaluator_override: Some(evaluator),
    })
    .unwrap();
    assert!(harness.run().await.is_err());
    assert_eq!(
        std::fs::read_to_string(project.path().join("greeting.txt")).unwrap(),
        "old\n"
    );
}

#[tokio::test]
async fn evaluation_evidence_is_bound_to_the_snapshot() {
    let project = tempfile::tempdir().unwrap();
    let run = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("greeting.txt"), "old\n").unwrap();
    let workspace = Workspace::create(project.path(), run.path()).unwrap();
    std::fs::write(workspace.stage_root().join("greeting.txt"), "new\n").unwrap();
    let snapshot = workspace.tree_hash().unwrap();
    let authorization = ApplyAuthorization {
        snapshot_hash: snapshot.clone(),
        evaluation_id: "eval_1".to_owned(),
        approved_at_ms: 0,
    };
    let evidence = ApplyEvidence {
        id: "eval_1".to_owned(),
        snapshot_hash: snapshot,
        policy_hash: "policy".to_owned(),
        passed: true,
    };

    // Change the reviewed bytes after approval: apply must refuse.
    std::fs::write(workspace.stage_root().join("greeting.txt"), "sneaky\n").unwrap();
    assert!(
        workspace
            .apply(run.path(), &authorization, &evidence)
            .is_err()
    );
    assert_eq!(
        std::fs::read_to_string(project.path().join("greeting.txt")).unwrap(),
        "old\n"
    );
}

#[tokio::test]
async fn ignores_cannot_hide_staged_changes() {
    let project = tempfile::tempdir().unwrap();
    let run = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("a.txt"), "a\n").unwrap();
    std::fs::write(project.path().join(".gitignore"), "*.txt\n").unwrap();
    let workspace = Workspace::create(project.path(), run.path()).unwrap();
    std::fs::write(workspace.stage_root().join("a.txt"), "changed\n").unwrap();
    std::fs::write(workspace.stage_root().join(".gitignore"), "nothing\n").unwrap();
    let diff = workspace.diff(1024 * 1024).unwrap();
    assert!(diff.changed.iter().any(|file| file.path == "a.txt"));
}

#[test]
fn scripted_provider_is_deterministic_and_never_networks() {
    let provider = ScriptedProvider::new(vec![text_response("one")]);
    let request = CompletionRequest {
        model: "m".to_owned(),
        system: String::new(),
        messages: vec![],
        tools: vec![],
        max_tokens: 1,
        temperature: 0.0,
    };
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let first = runtime
        .block_on(async { provider.complete(request.clone()).await })
        .unwrap();
    assert_eq!(first.text.as_deref(), Some("one"));
    assert!(
        runtime
            .block_on(async { provider.complete(request).await })
            .is_err()
    );
    assert_eq!(provider.requests().len(), 2);
}
