//! Typed orchestration: inspect → plan → approve → stage → execute →
//! evaluate → bounded repair → review → approve apply.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};

use crate::config::Config;
use crate::evaluator::{
    DockerEvaluator, EvaluationPolicy, EvaluationReport, Evaluator, failure_evidence,
};
use crate::providers::{ModelProvider, Usage, create_provider};
use crate::roles::executor::{self, ExecutionMode};
use crate::roles::planner::{self, Plan};
use crate::session::{
    BudgetLimits, BudgetSink, Phase, Role, SESSION_FORMAT_VERSION, SessionState, SessionStore,
};
use crate::tools::{DiffReport, ToolContext};
use crate::util::{atomic_write, hash_bytes, now_ms, redact, truncate_bytes};
use crate::workspace::{ApplyAuthorization, ApplyReport, Workspace, tree_hash};

#[derive(Debug, Clone)]
pub enum ApprovalRequest {
    Plan {
        plan: Plan,
        plan_hash: String,
    },
    Evaluation {
        policy: EvaluationPolicy,
        network_profiles: Vec<String>,
    },
    Apply {
        diff: DiffReport,
        snapshot_hash: String,
    },
}

#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    Approved,
    Rejected(String),
}

/// Approval is always external. A model can never satisfy this interface.
pub trait Approver: Send {
    fn approve(&mut self, request: ApprovalRequest) -> Result<ApprovalDecision>;
}

pub struct RejectAllApprover;

impl Approver for RejectAllApprover {
    fn approve(&mut self, _request: ApprovalRequest) -> Result<ApprovalDecision> {
        Ok(ApprovalDecision::Rejected(
            "non-interactive execution requires explicit approval flags".to_owned(),
        ))
    }
}

pub struct RunOutcome {
    pub session_id: String,
    pub phase: Phase,
    pub diff: Option<DiffReport>,
    pub evaluation: Option<EvaluationReport>,
    pub applied: Option<ApplyReport>,
    pub notes: Vec<String>,
}

pub struct HarnessBuilder {
    pub config: Config,
    pub project_root: PathBuf,
    pub task: String,
    pub approver: Box<dyn Approver>,
    pub resume_session: Option<String>,
    pub sessions_root: Option<PathBuf>,
    pub planner_override: Option<Arc<dyn ModelProvider>>,
    pub executor_override: Option<Arc<dyn ModelProvider>>,
    pub evaluator_override: Option<Arc<dyn Evaluator>>,
}

struct PersistentBudget {
    session_file: PathBuf,
    limits: BudgetLimits,
    state: Mutex<crate::session::BudgetState>,
}

impl PersistentBudget {
    fn new(
        session_file: PathBuf,
        limits: BudgetLimits,
        state: crate::session::BudgetState,
    ) -> Self {
        Self {
            session_file,
            limits,
            state: Mutex::new(state),
        }
    }

    fn snapshot(&self) -> Result<crate::session::BudgetState> {
        Ok(self.state.lock().expect("budget lock").clone())
    }

    fn update(
        &self,
        mutate: impl FnOnce(&mut crate::session::BudgetState) -> Result<()>,
    ) -> Result<()> {
        let mut state = self.state.lock().expect("budget lock");
        mutate(&mut state)?;
        let snapshot = state.clone();
        drop(state);
        // Persist the reservation before the caller performs the action.
        let text = std::fs::read_to_string(&self.session_file)
            .with_context(|| "failed to read session state for budget accounting")?;
        let mut session: SessionState =
            serde_json::from_str(&text).with_context(|| "session state is corrupt")?;
        session.budget = snapshot;
        session.updated_at_ms = now_ms();
        let data = serde_json::to_vec_pretty(&session)?;
        atomic_write(&self.session_file, &data, Some(0o600))
    }
}

impl BudgetSink for PersistentBudget {
    fn reserve_model(&self, role: Role, max_tokens: u32) -> Result<()> {
        self.update(|budget| {
            match role {
                Role::Planner => {
                    if budget.planner_calls >= self.limits.planner_calls {
                        bail!("planner model-call budget exhausted");
                    }
                    budget.planner_calls += 1;
                }
                Role::Executor => {
                    if budget.executor_calls >= self.limits.executor_calls {
                        bail!("executor model-call budget exhausted");
                    }
                    budget.executor_calls += 1;
                }
            }
            let next = budget.reserved_tokens.saturating_add(u64::from(max_tokens));
            if next > self.limits.tokens {
                bail!("run-wide model token budget exhausted");
            }
            budget.reserved_tokens = next;
            Ok(())
        })
    }

    fn reserve_tool(&self) -> Result<()> {
        self.update(|budget| {
            if budget.tool_calls >= self.limits.tool_calls {
                bail!("run-wide tool-call budget exhausted");
            }
            budget.tool_calls += 1;
            Ok(())
        })
    }

    fn record_usage(&self, usage: &Usage) -> Result<()> {
        self.update(|budget| {
            budget.input_tokens = budget
                .input_tokens
                .saturating_add(usage.input_tokens.unwrap_or(0));
            budget.output_tokens = budget
                .output_tokens
                .saturating_add(usage.output_tokens.unwrap_or(0));
            Ok(())
        })
    }
}

pub struct InMemoryBudget {
    limits: BudgetLimits,
    state: Mutex<crate::session::BudgetState>,
}

impl InMemoryBudget {
    pub fn new(limits: BudgetLimits) -> Self {
        Self {
            limits,
            state: Mutex::new(crate::session::BudgetState::default()),
        }
    }
}

impl BudgetSink for InMemoryBudget {
    fn reserve_model(&self, role: Role, max_tokens: u32) -> Result<()> {
        let mut budget = self.state.lock().expect("budget lock");
        match role {
            Role::Planner => {
                if budget.planner_calls >= self.limits.planner_calls {
                    bail!("planner model-call budget exhausted");
                }
                budget.planner_calls += 1;
            }
            Role::Executor => {
                if budget.executor_calls >= self.limits.executor_calls {
                    bail!("executor model-call budget exhausted");
                }
                budget.executor_calls += 1;
            }
        }
        budget.reserved_tokens = budget.reserved_tokens.saturating_add(u64::from(max_tokens));
        if budget.reserved_tokens > self.limits.tokens {
            bail!("run-wide model token budget exhausted");
        }
        Ok(())
    }

    fn reserve_tool(&self) -> Result<()> {
        let mut budget = self.state.lock().expect("budget lock");
        if budget.tool_calls >= self.limits.tool_calls {
            bail!("run-wide tool-call budget exhausted");
        }
        budget.tool_calls += 1;
        Ok(())
    }

    fn record_usage(&self, usage: &Usage) -> Result<()> {
        let mut budget = self.state.lock().expect("budget lock");
        budget.input_tokens = budget
            .input_tokens
            .saturating_add(usage.input_tokens.unwrap_or(0));
        budget.output_tokens = budget
            .output_tokens
            .saturating_add(usage.output_tokens.unwrap_or(0));
        Ok(())
    }
}

pub struct Harness {
    config: Config,
    secrets: Vec<String>,
    planner: Arc<dyn ModelProvider>,
    executor: Arc<dyn ModelProvider>,
    evaluator: Arc<dyn Evaluator>,
    store: SessionStore,
    state: SessionState,
    budget: Arc<PersistentBudget>,
    approver: Box<dyn Approver>,
    workspace: Option<Workspace>,
}

impl Harness {
    pub fn create(builder: HarnessBuilder) -> Result<Self> {
        let config = builder.config;
        config.validate()?;
        let secrets = config.secret_values();
        let project_root = builder.project_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve project root {}",
                builder.project_root.display()
            )
        })?;
        let sessions_root = match builder.sessions_root {
            Some(path) => path,
            None => config.sessions_dir()?,
        };
        let (store, state) = if let Some(id) = builder.resume_session {
            let (store, mut state) = SessionStore::open(&sessions_root, &id)?;
            let notes = store.reconcile(&mut state)?;
            for note in &notes {
                store.event("resume", serde_json::json!({"note": note}), &secrets)?;
            }
            if state.project_root != project_root {
                bail!("session belongs to a different project");
            }
            if state.version != SESSION_FORMAT_VERSION {
                bail!("session version is not supported");
            }
            store.save(&mut state)?;
            (store, state)
        } else {
            SessionStore::create(&sessions_root, &project_root, &builder.task)?
        };
        let limits = BudgetLimits {
            planner_calls: u32::from(config.harness.max_planner_calls),
            executor_calls: u32::from(config.harness.max_executor_calls),
            tool_calls: config.harness.max_tool_calls as u32,
            tokens: config.harness.max_model_tokens,
            context_tokens: config.harness.max_context_tokens,
        };
        let budget = Arc::new(PersistentBudget::new(
            store.state_path(),
            limits,
            state.budget.clone(),
        ));
        let planner = match builder.planner_override {
            Some(provider) => provider,
            None => create_provider(
                config.roles.planner.provider,
                config.provider_settings(config.roles.planner.provider)?,
            )?,
        };
        let executor = match builder.executor_override {
            Some(provider) => provider,
            None => create_provider(
                config.roles.executor.provider,
                config.provider_settings(config.roles.executor.provider)?,
            )?,
        };
        let evaluator: Arc<dyn Evaluator> = match builder.evaluator_override {
            Some(evaluator) => evaluator,
            None => Arc::new(DockerEvaluator::new(config.docker.clone())),
        };
        let workspace = if state.run_dir.join("workspace.json").exists() {
            Some(Workspace::open(&state.run_dir)?)
        } else {
            None
        };
        Ok(Self {
            config,
            secrets,
            planner,
            executor,
            evaluator,
            store,
            state,
            budget,
            approver: builder.approver,
            workspace,
        })
    }

    pub fn session_id(&self) -> &str {
        &self.state.id
    }

    pub async fn run(mut self) -> Result<RunOutcome> {
        let result = match self.state.phase {
            Phase::Created => self.run_new().await,
            Phase::PlanReview
            | Phase::Staging
            | Phase::Review
            | Phase::ApplyReview
            | Phase::Evaluating => self.resume().await,
            Phase::NeedsAttention => bail!(
                "session needs attention: {}",
                self.state
                    .error
                    .clone()
                    .unwrap_or_else(|| "interrupted work must be inspected".to_owned())
            ),
            Phase::Failed => bail!(
                "session failed: {}",
                self.state
                    .error
                    .clone()
                    .unwrap_or_else(|| "no further work".to_owned())
            ),
            phase => bail!("session is already in phase {}", phase.as_str()),
        };
        if let Err(error) = &result
            && self.state.phase != Phase::NeedsAttention
            && self.state.phase != Phase::Failed
        {
            // Unexpected errors (provider/disk/tool) must be recorded rather
            // than leaving a misleading in-flight phase behind.
            self.state.phase = Phase::Failed;
            self.state.error = Some(redact(&format!("{error:#}"), &self.secrets));
            self.state.budget = self.budget.snapshot()?;
            self.store.save(&mut self.state)?;
            self.event(
                "failed",
                serde_json::json!({"message": format!("{error:#}")}),
            )?;
        }
        result
    }

    async fn run_new(&mut self) -> Result<RunOutcome> {
        self.set_phase(Phase::Planning)?;
        self.event(
            "planning",
            serde_json::json!({"task": self.state.task.clone()}),
        )?;
        let planning_context = ToolContext::new(
            &self.state.project_root,
            None,
            self.config.harness.max_read_bytes,
            self.config.harness.max_tool_output_bytes,
        )?;
        let test_names: Vec<String> = self
            .config
            .tests
            .iter()
            .map(|test| test.name.clone())
            .collect();
        let planning = planner::create_plan(
            self.planner.as_ref(),
            &self.config.roles.planner,
            &planning_context,
            &self.state.task,
            &test_names,
            &self.secrets,
            self.budget.as_ref(),
            self.config.harness.max_context_tokens,
        )
        .await?;
        let plan_hash = hash_bytes(serde_json::to_vec(&planning.plan)?.as_slice());
        self.state.plan = Some(planning.plan.clone());
        self.state.plan_hash = Some(plan_hash.clone());
        self.set_phase(Phase::PlanReview)?;
        self.event(
            "plan",
            serde_json::json!({"hash": plan_hash, "model_calls": planning.model_calls}),
        )?;
        self.require_plan_approval(&planning.plan, &plan_hash)?;
        self.after_plan_approval().await
    }

    async fn resume(&mut self) -> Result<RunOutcome> {
        let plan = self
            .state
            .plan
            .clone()
            .context("cannot resume without a saved plan")?;
        let plan_hash = self
            .state
            .plan_hash
            .clone()
            .context("cannot resume without a plan hash")?;
        let recomputed = hash_bytes(serde_json::to_vec(&plan)?.as_slice());
        if recomputed != plan_hash {
            bail!("saved plan does not match its recorded hash; refusing to resume");
        }
        if !self.state.approval("plan", &plan_hash) {
            self.require_plan_approval(&plan, &plan_hash)?;
        }
        if self.state.phase == Phase::PlanReview || self.state.phase == Phase::Staging {
            return self.after_plan_approval().await;
        }
        self.workspace = Some(Workspace::open(&self.state.run_dir)?);
        if self.state.phase == Phase::Review || self.state.phase == Phase::ApplyReview {
            return self.review_and_apply().await;
        }
        if self.state.phase == Phase::Evaluating {
            return self.evaluate_and_repair(&plan).await;
        }
        bail!(
            "session is not resumable from phase {}",
            self.state.phase.as_str()
        )
    }

    fn require_plan_approval(&mut self, plan: &Plan, plan_hash: &str) -> Result<()> {
        let decision = self.approver.approve(ApprovalRequest::Plan {
            plan: plan.clone(),
            plan_hash: plan_hash.to_owned(),
        })?;
        match decision {
            ApprovalDecision::Approved => {
                self.state.record_approval("plan", plan_hash);
                self.store.save(&mut self.state)?;
                Ok(())
            }
            ApprovalDecision::Rejected(reason) => self.fail(format!("plan rejected: {reason}")),
        }
    }

    async fn after_plan_approval(&mut self) -> Result<RunOutcome> {
        let plan = self.state.plan.clone().context("missing approved plan")?;
        if self.workspace.is_none() {
            self.set_phase(Phase::Staging)?;
            self.workspace = Some(Workspace::create(
                &self.state.project_root,
                &self.state.run_dir,
            )?);
        }
        let workspace = self.workspace.clone().context("workspace missing")?;
        let scope: Vec<String> = plan.files.iter().map(|file| file.path.clone()).collect();
        let context = ToolContext::new(
            workspace.stage_root(),
            Some(workspace.baseline_root().to_path_buf()),
            self.config.harness.max_read_bytes,
            self.config.harness.max_tool_output_bytes,
        )?
        .with_scope(scope);
        let test_names: Vec<String> = self
            .config
            .tests
            .iter()
            .map(|test| test.name.clone())
            .collect();
        let snapshot = workspace.tree_hash()?;
        self.set_phase(Phase::Executing)?;
        self.event(
            "execute",
            serde_json::json!({"repair": false, "snapshot": snapshot}),
        )?;
        let result = executor::execute(
            self.executor.as_ref(),
            &self.config.roles.executor,
            &context,
            &plan,
            &self.state.task,
            &ExecutionMode::Initial,
            &test_names,
            &self.secrets,
            self.budget.as_ref(),
            self.config.harness.max_context_tokens,
        )
        .await?;
        self.event(
            "executor_summary",
            serde_json::json!({"summary": truncate_bytes(&result.summary, 4000)}),
        )?;
        self.evaluate_and_repair(&plan).await
    }

    async fn evaluate_and_repair(&mut self, plan: &Plan) -> Result<RunOutcome> {
        if self.config.tests.is_empty() {
            return self.fail(
                "no test profiles are configured; evaluation fails closed because the model cannot verify its own work"
                    .to_owned(),
            );
        }
        let policy = EvaluationPolicy::from_tests(&self.config.tests);
        policy.validate()?;
        let policy_hash = policy.hash();
        if !self.state.approval("evaluation", &policy_hash) {
            let network: Vec<String> = policy
                .profiles
                .iter()
                .filter(|profile| profile.allow_network || profile.prepare_allow_network)
                .map(|profile| profile.name.clone())
                .collect();
            let decision = self.approver.approve(ApprovalRequest::Evaluation {
                policy: policy.clone(),
                network_profiles: network,
            })?;
            match decision {
                ApprovalDecision::Approved => {
                    self.state.record_approval("evaluation", &policy_hash);
                    self.store.save(&mut self.state)?;
                }
                ApprovalDecision::Rejected(reason) => {
                    return self.fail(format!("evaluation rejected: {reason}"));
                }
            }
        }

        let max_repairs = self.config.harness.max_repairs;
        loop {
            self.set_phase(Phase::Evaluating)?;
            let workspace = self.workspace.clone().context("workspace missing")?;
            let evaluation = self
                .evaluator
                .evaluate(&workspace, &policy, &self.state.run_dir, &self.state.id)
                .await?;
            self.write_evaluation(&evaluation)?;
            self.state.evaluation = Some(crate::session::EvaluationSummary {
                id: evaluation.id.clone(),
                snapshot_hash: evaluation.snapshot_hash.clone(),
                policy_hash: evaluation.policy_hash.clone(),
                passed: evaluation.all_passed,
                summary: crate::evaluator::summary(&evaluation.outcomes),
            });
            self.store.save(&mut self.state)?;
            self.event(
                "evaluation",
                serde_json::json!({
                    "id": evaluation.id,
                    "passed": evaluation.all_passed,
                    "summary": crate::evaluator::summary(&evaluation.outcomes),
                }),
            )?;
            let setup_failed = evaluation
                .outcomes
                .iter()
                .any(|outcome| outcome.setup_failure.is_some());
            let passed = evaluation.all_passed;
            let failures = failure_evidence(
                &evaluation.outcomes,
                self.config.harness.max_tool_output_bytes,
            );
            if passed {
                break;
            }
            if setup_failed {
                return self.fail(format!(
                    "evaluation failed for infrastructure or setup reasons, not a repairable test failure:\n{failures}"
                ));
            }
            if self.state.budget.repairs >= max_repairs {
                return self.fail(format!(
                    "repair budget exhausted after {} attempt(s):\n{failures}",
                    self.state.budget.repairs
                ));
            }
            self.budget.update_repairs()?;
            self.set_phase(Phase::Repairing)?;
            let evidence = failures;
            let scope: Vec<String> = plan.files.iter().map(|file| file.path.clone()).collect();
            let workspace = self.workspace.clone().context("workspace missing")?;
            let context = ToolContext::new(
                workspace.stage_root(),
                Some(workspace.baseline_root().to_path_buf()),
                self.config.harness.max_read_bytes,
                self.config.harness.max_tool_output_bytes,
            )?
            .with_scope(scope);
            let test_names: Vec<String> = self
                .config
                .tests
                .iter()
                .map(|test| test.name.clone())
                .collect();
            self.event(
                "repair",
                serde_json::json!({"attempt": self.state.budget.repairs}),
            )?;
            let result = executor::execute(
                self.executor.as_ref(),
                &self.config.roles.executor,
                &context,
                plan,
                &self.state.task,
                &ExecutionMode::Repair { evidence },
                &test_names,
                &self.secrets,
                self.budget.as_ref(),
                self.config.harness.max_context_tokens,
            )
            .await?;
            self.event(
                "executor_summary",
                serde_json::json!({"summary": truncate_bytes(&result.summary, 4000)}),
            )?;
        }

        self.review_and_apply().await
    }

    async fn review_and_apply(&mut self) -> Result<RunOutcome> {
        let workspace = self.workspace.clone().context("workspace missing")?;
        let diff = workspace.diff(self.config.harness.max_tool_output_bytes)?;
        let snapshot_hash = tree_hash(&workspace.current_files()?)?;
        let evaluation = self
            .state
            .evaluation
            .clone()
            .context("no evaluation evidence for this snapshot")?;
        if evaluation.snapshot_hash != snapshot_hash {
            return self.fail(
                "the staged snapshot changed after evaluation; re-evaluate before applying"
                    .to_owned(),
            );
        }
        if !evaluation.passed {
            return self.fail("cannot apply: evaluation did not pass".to_owned());
        }
        self.set_phase(Phase::ApplyReview)?;
        if !self.state.approval("apply", &snapshot_hash) {
            let decision = self.approver.approve(ApprovalRequest::Apply {
                diff: diff.clone(),
                snapshot_hash: snapshot_hash.clone(),
            })?;
            match decision {
                ApprovalDecision::Approved => {
                    self.state.record_approval("apply", &snapshot_hash);
                    self.store.save(&mut self.state)?;
                }
                ApprovalDecision::Rejected(reason) => {
                    return self.fail(format!("apply rejected: {reason}"));
                }
            }
        }
        let report = self.read_evaluation()?;
        if report.snapshot_hash != snapshot_hash || !report.all_passed {
            return self.fail("stored evaluation evidence is stale; re-evaluate".to_owned());
        }
        let authorization = ApplyAuthorization {
            snapshot_hash: snapshot_hash.clone(),
            evaluation_id: report.id.clone(),
            approved_at_ms: now_ms(),
        };
        let outcome = workspace.apply(&self.state.run_dir, &authorization, &report.evidence());
        let applied = match outcome {
            Ok(applied) => applied,
            Err(error) => {
                self.state.phase = Phase::NeedsAttention;
                self.state.error = Some(format!("{error:#}"));
                self.store.save(&mut self.state)?;
                return Err(error);
            }
        };
        self.state.apply = Some(crate::session::ApplySummary {
            snapshot_hash,
            evaluation_id: report.id.clone(),
            state: "applied".to_owned(),
            journal: applied.journal.clone(),
            rollback_errors: Vec::new(),
        });
        self.set_phase(Phase::Completed)?;
        self.event(
            "applied",
            serde_json::json!({"files": applied.applied.len(), "journal": applied.journal}),
        )?;
        Ok(RunOutcome {
            session_id: self.state.id.clone(),
            phase: Phase::Completed,
            diff: Some(diff),
            evaluation: Some(report),
            applied: Some(applied),
            notes: Vec::new(),
        })
    }

    fn write_evaluation(&self, report: &EvaluationReport) -> Result<()> {
        let data = serde_json::to_vec_pretty(report)?;
        atomic_write(
            &self.state.run_dir.join("evaluation.json"),
            &data,
            Some(0o600),
        )
    }

    fn read_evaluation(&self) -> Result<EvaluationReport> {
        let text = std::fs::read_to_string(self.state.run_dir.join("evaluation.json"))
            .context("evaluation evidence is missing")?;
        serde_json::from_str(&text).context("evaluation evidence is corrupt")
    }

    fn set_phase(&mut self, phase: Phase) -> Result<()> {
        self.state.phase = phase;
        self.state.budget = self.budget.snapshot()?;
        self.store.save(&mut self.state)?;
        Ok(())
    }

    fn event(&self, kind: &str, details: serde_json::Value) -> Result<()> {
        self.store.event(kind, details, &self.secrets)
    }

    fn fail<T>(&mut self, message: String) -> Result<T> {
        self.state.error = Some(redact(&message, &self.secrets));
        self.state.phase = Phase::Failed;
        self.state.budget = self.budget.snapshot()?;
        self.store.save(&mut self.state)?;
        self.event("failed", serde_json::json!({"message": message}))?;
        Err(anyhow::anyhow!(message))
    }
}

impl PersistentBudget {
    fn update_repairs(&self) -> Result<()> {
        self.update(|budget| {
            budget.repairs = budget.repairs.saturating_add(1);
            Ok(())
        })
    }
}
