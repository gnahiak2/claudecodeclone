use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand};
use relay_code::config::{Config, ProviderKind};
use relay_code::harness::{
    ApprovalDecision, ApprovalRequest, Approver, Harness, HarnessBuilder, InMemoryBudget,
};
use relay_code::providers::{Message, create_provider};
use relay_code::roles::chat;
use relay_code::roles::planner;
use relay_code::session::{BudgetLimits, SessionStore};
use relay_code::tools::ToolContext;
use relay_code::util::sanitize_terminal;

#[derive(Parser)]
#[command(
    name = "relay",
    version,
    about = "Three-tier coding agent: read-only planning, staged execution, isolated evaluation"
)]
struct Cli {
    /// Configuration file. Hosted providers require credentials in environment
    /// variables named by the configuration.
    #[arg(
        long,
        global = true,
        env = "RELAY_CONFIG",
        default_value = "relay.toml"
    )]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read-only conversation about the repository.
    Chat,
    /// Produce a read-only implementation plan. Never modifies files.
    Plan {
        /// Task description.
        task: String,
    },
    /// Run the full plan → execute → evaluate → review → apply pipeline.
    Run(RunArgs),
    /// Inspect configured providers and role models (never prints secrets).
    Providers,
    /// Session inspection and resume.
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
}

#[derive(Args)]
struct RunArgs {
    /// Task description. Ignored when resuming a session.
    #[arg(default_value = "")]
    task: String,
    /// Approve the plan without prompting.
    #[arg(long)]
    approve_plan: bool,
    /// Approve the exact evaluation policy without prompting.
    #[arg(long)]
    approve_evaluation: bool,
    /// Approve the reviewed diff and apply without prompting.
    #[arg(long)]
    approve_apply: bool,
    /// Never prompt; unapproved steps fail safely.
    #[arg(long)]
    non_interactive: bool,
    /// Resume an existing session instead of starting a new one.
    #[arg(long)]
    resume: Option<String>,
}

#[derive(Subcommand)]
enum SessionCommand {
    /// Print a session's state and recent events.
    Inspect { id: String },
    /// Resume a session from a safe checkpoint.
    Resume {
        id: String,
        #[arg(long)]
        approve_plan: bool,
        #[arg(long)]
        approve_evaluation: bool,
        #[arg(long)]
        approve_apply: bool,
        #[arg(long)]
        non_interactive: bool,
    },
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {}", sanitize_terminal(&format!("{error:#}")));
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;
    match cli.command {
        Command::Providers => providers(&config, cli.config.as_path()),
        Command::Plan { task } => plan(&config, &task).await,
        Command::Chat => chat_mode(&config).await,
        Command::Run(args) => {
            let approver = CliApprover::new(
                !args.non_interactive,
                AutoApprove {
                    plan: args.approve_plan,
                    evaluation: args.approve_evaluation,
                    apply: args.approve_apply,
                },
            );
            let task = if args.task.trim().is_empty() {
                "Implement the requested task.".to_owned()
            } else {
                args.task.clone()
            };
            run_harness(config, task, approver, args.resume).await
        }
        Command::Session { command } => match command {
            SessionCommand::Inspect { id } => inspect_session(&config, &id),
            SessionCommand::Resume {
                id,
                approve_plan,
                approve_evaluation,
                approve_apply,
                non_interactive,
            } => {
                let approver = CliApprover::new(
                    !non_interactive,
                    AutoApprove {
                        plan: approve_plan,
                        evaluation: approve_evaluation,
                        apply: approve_apply,
                    },
                );
                run_harness(config, String::new(), approver, Some(id)).await
            }
        },
    }
}

fn providers(config: &Config, path: &std::path::Path) -> Result<()> {
    println!("config: {}", sanitize_terminal(&path.display().to_string()));
    let roles = [
        ("planner", &config.roles.planner),
        ("executor", &config.roles.executor),
    ];
    for (name, role) in roles {
        let settings = config.providers.get(role.provider);
        let key_state = match settings.and_then(|settings| settings.api_key_env.as_deref()) {
            Some(variable) => {
                if std::env::var_os(variable).is_some() {
                    format!("{variable}=set")
                } else {
                    format!("{variable}=unset")
                }
            }
            None => "no api key".to_owned(),
        };
        println!(
            "{name}: provider={} model={} tools={} ({key_state})",
            role.provider.as_str(),
            sanitize_terminal(&role.model),
            settings.map(|settings| settings.tools).unwrap_or(false),
        );
    }
    for kind in [
        ProviderKind::Ollama,
        ProviderKind::Cohere,
        ProviderKind::Openai,
        ProviderKind::Anthropic,
        ProviderKind::Compatible,
    ] {
        if let Some(settings) = config.providers.get(kind) {
            let endpoint = settings
                .base_url
                .as_deref()
                .unwrap_or_else(|| default_endpoint(kind));
            println!(
                "{}: endpoint={} timeout={}s trust_custom_endpoint={}",
                kind.as_str(),
                sanitize_terminal(endpoint),
                settings.timeout_secs.unwrap_or(120),
                settings.trust_custom_endpoint
            );
        }
    }
    println!(
        "docker: runtime={} image_policy=explicitly-pulled-only network=disabled-by-default",
        config.docker.runtime
    );
    Ok(())
}

fn default_endpoint(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Ollama => "http://localhost:11434",
        ProviderKind::Cohere => "https://api.cohere.com",
        ProviderKind::Openai => "https://api.openai.com/v1",
        ProviderKind::Anthropic => "https://api.anthropic.com",
        ProviderKind::Compatible => "(required)",
    }
}

fn limits(config: &Config) -> BudgetLimits {
    BudgetLimits {
        planner_calls: u32::from(config.harness.max_planner_calls),
        executor_calls: u32::from(config.harness.max_executor_calls),
        tool_calls: config.harness.max_tool_calls.min(u32::MAX as usize) as u32,
        tokens: config.harness.max_model_tokens,
        context_tokens: config.harness.max_context_tokens,
    }
}

async fn plan(config: &Config, task: &str) -> Result<()> {
    if task.trim().is_empty() {
        bail!("task must not be empty");
    }
    let project = std::env::current_dir()?.canonicalize()?;
    let provider = create_provider(
        config.roles.planner.provider,
        config.provider_settings(config.roles.planner.provider)?,
    )?;
    let context = ToolContext::new(
        &project,
        None,
        config.harness.max_read_bytes,
        config.harness.max_tool_output_bytes,
    )?;
    let budget = InMemoryBudget::new(limits(config));
    let test_names: Vec<String> = config.tests.iter().map(|test| test.name.clone()).collect();
    let result = planner::create_plan(
        provider.as_ref(),
        &config.roles.planner,
        &context,
        task,
        &test_names,
        &config.secret_values(),
        &budget,
        config.harness.max_context_tokens,
    )
    .await?;
    println!("{}", sanitize_terminal(&result.plan.display()));
    println!(
        "\n(planner model calls: {}, tool calls: {})",
        result.model_calls, result.tool_calls
    );
    Ok(())
}

async fn chat_mode(config: &Config) -> Result<()> {
    let project = std::env::current_dir()?.canonicalize()?;
    let provider = create_provider(
        config.roles.planner.provider,
        config.provider_settings(config.roles.planner.provider)?,
    )?;
    let context = ToolContext::new(
        &project,
        None,
        config.harness.max_read_bytes,
        config.harness.max_tool_output_bytes,
    )?;
    let budget = InMemoryBudget::new(limits(config));
    let secrets = config.secret_values();
    let mut messages: Vec<Message> = Vec::new();
    let stdin = std::io::stdin();
    let interactive = stdin.is_terminal();
    if interactive {
        println!("read-only chat. type 'exit' to quit.");
    }
    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, "exit" | "quit") {
            break;
        }
        messages.push(Message::User {
            content: trimmed.to_owned(),
        });
        let answer = chat::respond(
            provider.as_ref(),
            &config.roles.planner,
            &context,
            &mut messages,
            &secrets,
            &budget,
            config.harness.max_context_tokens,
        )
        .await?;
        println!("{}", sanitize_terminal(answer.trim()));
    }
    Ok(())
}

async fn run_harness(
    config: Config,
    task: String,
    approver: CliApprover,
    resume: Option<String>,
) -> Result<()> {
    let project = std::env::current_dir()?.canonicalize()?;
    let harness = Harness::create(HarnessBuilder {
        config,
        project_root: project,
        task,
        approver: Box::new(approver),
        resume_session: resume,
        sessions_root: None,
        planner_override: None,
        executor_override: None,
        evaluator_override: None,
    })?;
    let session_id = harness.session_id().to_owned();
    println!("session: {session_id}");
    match harness.run().await {
        Ok(outcome) => {
            if let Some(diff) = &outcome.diff {
                for file in &diff.changed {
                    println!("{:?} {}", file.status, sanitize_terminal(&file.path));
                }
            }
            if let Some(applied) = &outcome.applied {
                println!(
                    "applied {} file(s); journal: {}",
                    applied.applied.len(),
                    sanitize_terminal(&applied.journal.display().to_string())
                );
            }
            println!("phase: {}", outcome.phase.as_str());
            Ok(())
        }
        Err(error) => {
            let message = sanitize_terminal(&format!("{error:#}"));
            eprintln!("run failed: {message}");
            eprintln!("session preserved: {session_id}");
            Err(anyhow::anyhow!("run failed"))
        }
    }
}

fn inspect_session(config: &Config, id: &str) -> Result<()> {
    let root = config.sessions_dir()?;
    let (store, state) = SessionStore::open(&root, id)?;
    println!("session: {}", state.id);
    println!("phase: {}", state.phase.as_str());
    println!(
        "project: {}",
        sanitize_terminal(&state.project_root.display().to_string())
    );
    println!("task: {}", sanitize_terminal(&state.task));
    println!(
        "budget: planner={} executor={} tools={} reserved_tokens={} input_tokens={} output_tokens={} repairs={}",
        state.budget.planner_calls,
        state.budget.executor_calls,
        state.budget.tool_calls,
        state.budget.reserved_tokens,
        state.budget.input_tokens,
        state.budget.output_tokens,
        state.budget.repairs
    );
    if let Some(plan) = &state.plan {
        println!("\n{}", sanitize_terminal(&plan.display()));
    }
    if let Some(evaluation) = &state.evaluation {
        println!(
            "evaluation: {} passed={} snapshot={}",
            evaluation.id, evaluation.passed, evaluation.snapshot_hash
        );
    }
    if let Some(apply) = &state.apply {
        println!(
            "apply: state={} journal={}",
            apply.state,
            sanitize_terminal(&apply.journal.display().to_string())
        );
    }
    if let Some(error) = &state.error {
        println!("error: {}", sanitize_terminal(error));
    }
    let events_path = store.directory().join("events.jsonl");
    if let Ok(text) = std::fs::read_to_string(events_path) {
        let lines: Vec<&str> = text.lines().collect();
        let start = lines.len().saturating_sub(10);
        println!("\nrecent events:");
        for line in &lines[start..] {
            println!("{}", sanitize_terminal(line));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AutoApprove {
    plan: bool,
    evaluation: bool,
    apply: bool,
}

struct CliApprover {
    interactive: bool,
    auto: AutoApprove,
}

impl CliApprover {
    fn new(interactive: bool, auto: AutoApprove) -> Self {
        Self {
            interactive: interactive && std::io::stdin().is_terminal(),
            auto,
        }
    }
}

impl Approver for CliApprover {
    fn approve(&mut self, request: ApprovalRequest) -> Result<ApprovalDecision> {
        let (auto, description) = match &request {
            ApprovalRequest::Plan { plan, .. } => (self.auto.plan, plan.display()),
            ApprovalRequest::Evaluation {
                policy,
                network_profiles,
            } => {
                let mut text = String::new();
                for profile in &policy.profiles {
                    text.push_str(&format!(
                        "profile {}: image={} command={:?} cwd={} timeout={}s network={} prepare_network={}\n",
                        profile.name,
                        profile.image,
                        profile.command,
                        profile.cwd,
                        profile.timeout_secs,
                        profile.allow_network,
                        profile.prepare_allow_network
                    ));
                }
                if !network_profiles.is_empty() {
                    text.push_str(&format!(
                        "network-enabled profiles: {}\n",
                        network_profiles.join(", ")
                    ));
                }
                text.push_str(
                    "sandbox: isolated copy, non-root, read-only root, dropped capabilities, no-new-privileges, CPU/memory/PID limits, no host credentials or docker socket\n",
                );
                (self.auto.evaluation, text)
            }
            ApprovalRequest::Apply { diff, .. } => {
                let mut text = String::new();
                for file in &diff.changed {
                    text.push_str(&format!("{:?} {}\n", file.status, file.path));
                }
                text.push('\n');
                text.push_str(&diff.patch);
                (self.auto.apply, text)
            }
        };
        println!(
            "\n--- approval required ---\n{}",
            sanitize_terminal(&description)
        );
        if auto {
            println!("approved by explicit command-line flag");
            return Ok(ApprovalDecision::Approved);
        }
        if !self.interactive {
            return Ok(ApprovalDecision::Rejected(
                "approval required but stdin is not an interactive terminal".to_owned(),
            ));
        }
        print!("approve? [y/N] ");
        std::io::stdout().flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            Ok(ApprovalDecision::Approved)
        } else {
            Ok(ApprovalDecision::Rejected("declined at prompt".to_owned()))
        }
    }
}
