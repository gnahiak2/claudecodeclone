use std::process::Command;

fn relay() -> Command {
    Command::new(env!("CARGO_BIN_EXE_relay"))
}

#[test]
fn help_lists_the_documented_commands() {
    let output = relay().arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for command in ["chat", "plan", "run", "providers", "session"] {
        assert!(text.contains(command), "missing {command} in {text}");
    }
}

#[test]
fn providers_prints_configuration_without_secret_values() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("relay.toml");
    std::fs::write(
        &config,
        r#"
[roles.planner]
provider = "compatible"
model = "local-model"

[roles.executor]
provider = "ollama"
model = "local-model"

[providers.compatible]
base_url = "https://example.invalid/v1"
api_key_env = "RELAY_TEST_SECRET"
trust_custom_endpoint = true

[providers.ollama]
base_url = "http://localhost:11434"
"#,
    )
    .unwrap();
    let output = relay()
        .current_dir(directory.path())
        .arg("--config")
        .arg(&config)
        .arg("providers")
        .env("RELAY_TEST_SECRET", "super-secret-value")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("RELAY_TEST_SECRET=set"));
    assert!(!text.contains("super-secret-value"));
    assert!(text.contains("network=disabled-by-default"));
}

#[test]
fn unknown_session_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("relay.toml");
    std::fs::write(
        &config,
        r#"
[harness]
sessions_dir = "/tmp/relay-cli-test-sessions"

[roles.planner]
provider = "ollama"
model = "local-model"

[roles.executor]
provider = "ollama"
model = "local-model"

[providers.ollama]
base_url = "http://localhost:11434"
"#,
    )
    .unwrap();
    let output = relay()
        .current_dir(directory.path())
        .arg("--config")
        .arg(&config)
        .args(["session", "inspect", "ses_doesnotexist"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains("does not exist") || text.contains("error"),
        "{text}"
    );
}
