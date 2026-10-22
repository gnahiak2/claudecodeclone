//! Docker integration tests. They are ignored by default because Docker is not
//! available in every environment and images must be pulled explicitly.
//!
//! Enable with:
//!   RELAY_DOCKER_TESTS=1 RELAY_TEST_IMAGE=alpine:3.20 \
//!     cargo test --test docker_evaluator -- --ignored --test-threads=1

use relay_code::config::DockerConfig;
use relay_code::evaluator::{
    ApprovedProfile, DockerEvaluator, EvaluationPolicy, Evaluator, TestOutcome,
};
use relay_code::workspace::Workspace;

fn enabled() -> bool {
    std::env::var("RELAY_DOCKER_TESTS").as_deref() == Ok("1")
}

fn image() -> String {
    std::env::var("RELAY_TEST_IMAGE").unwrap_or_else(|_| "alpine:3.20".to_owned())
}

fn profile(command: Vec<&str>, timeout_secs: u64, network: bool) -> ApprovedProfile {
    ApprovedProfile {
        name: "integration".to_owned(),
        image: image(),
        command: command.into_iter().map(str::to_owned).collect(),
        cwd: ".".to_owned(),
        timeout_secs,
        allow_network: network,
        prepare_command: vec![],
        prepare_allow_network: false,
    }
}

async fn evaluate(command: Vec<&str>, timeout_secs: u64, network: bool) -> TestOutcome {
    if !enabled() {
        panic!("set RELAY_DOCKER_TESTS=1 to run Docker integration tests");
    }
    let project = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("data.txt"), "sandbox\n").unwrap();
    let run = tempfile::tempdir().unwrap();
    let workspace = Workspace::create(project.path(), run.path()).unwrap();
    let policy = EvaluationPolicy {
        profiles: vec![profile(command, timeout_secs, network)],
    };
    let evaluator = DockerEvaluator::new(DockerConfig {
        output_limit_bytes: 64 * 1024,
        ..DockerConfig::default()
    });
    let report = evaluator
        .evaluate(&workspace, &policy, run.path(), "docker-test")
        .await
        .unwrap();
    report.outcomes.into_iter().next().unwrap()
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn passes_a_successful_command() {
    let outcome = evaluate(vec!["sh", "-c", "true"], 30, false).await;
    assert!(outcome.passed, "{:?}", outcome.result);
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn reports_a_failed_command() {
    let outcome = evaluate(vec!["sh", "-c", "exit 3"], 30, false).await;
    assert!(!outcome.passed);
    assert_eq!(outcome.result.unwrap().exit_code, Some(3));
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn enforces_the_wall_clock_timeout() {
    let outcome = evaluate(vec!["sh", "-c", "sleep 30"], 1, false).await;
    assert!(!outcome.passed);
    assert!(outcome.result.unwrap().timed_out);
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn network_is_disabled_by_default() {
    let outcome = evaluate(
        vec![
            "sh",
            "-c",
            "wget -q -T 3 -O /dev/null http://example.com || exit 7",
        ],
        20,
        false,
    )
    .await;
    assert!(!outcome.passed);
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn host_credentials_are_not_inherited() {
    unsafe {
        std::env::set_var("OPENAI_API_KEY", "should-not-leak");
    }
    let outcome = evaluate(
        vec![
            "sh",
            "-c",
            "test -z \"$OPENAI_API_KEY\" && test -z \"$ANTHROPIC_API_KEY\"",
        ],
        30,
        false,
    )
    .await;
    assert!(outcome.passed, "{:?}", outcome.result);
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn runs_as_non_root_with_a_read_only_root_filesystem() {
    let outcome = evaluate(
        vec![
            "sh",
            "-c",
            "test \"$(id -u)\" = 65534 && ! touch /relay-root-write",
        ],
        30,
        false,
    )
    .await;
    assert!(outcome.passed, "{:?}", outcome.result);
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn output_is_bounded() {
    let outcome = evaluate(vec!["sh", "-c", "yes relay | head -c 10000000"], 30, false).await;
    let result = outcome.result.unwrap();
    assert!(result.stdout_truncated || result.stderr_truncated);
    assert!(result.stdout.len() <= 64 * 1024);
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn cleanup_runs_after_failure() {
    let outcome = evaluate(vec!["sh", "-c", "exit 9"], 30, false).await;
    assert!(outcome.cleanup_error.is_none());
    assert!(!outcome.passed);
    // Best-effort check that no relay container for this run remains.
    let output = std::process::Command::new("docker")
        .args([
            "ps",
            "-a",
            "--filter",
            "name=relay-",
            "--format",
            "{{.Names}}",
        ])
        .output()
        .unwrap();
    let names = String::from_utf8_lossy(&output.stdout);
    assert!(!names.contains("relay-dockertest"), "{names}");
}

#[tokio::test]
#[ignore = "requires Docker and an explicitly pulled image"]
async fn missing_images_fail_closed() {
    let project = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("a.txt"), "a").unwrap();
    let run = tempfile::tempdir().unwrap();
    let workspace = Workspace::create(project.path(), run.path()).unwrap();
    let policy = EvaluationPolicy {
        profiles: vec![ApprovedProfile {
            name: "missing".to_owned(),
            image: "relay-definitely-missing-image:never".to_owned(),
            command: vec!["true".to_owned()],
            cwd: ".".to_owned(),
            timeout_secs: 10,
            allow_network: false,
            prepare_command: vec![],
            prepare_allow_network: false,
        }],
    };
    let evaluator = DockerEvaluator::new(DockerConfig::default());
    let error = evaluator
        .evaluate(&workspace, &policy, run.path(), "missing")
        .await
        .unwrap_err();
    assert!(format!("{error:#}").contains("not available locally"));
}
