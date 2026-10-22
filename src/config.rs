use std::collections::BTreeSet;
use std::env;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub harness: HarnessConfig,
    #[serde(default)]
    pub docker: DockerConfig,
    pub roles: RolesConfig,
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub tests: Vec<TestProfile>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct HarnessConfig {
    pub sessions_dir: Option<PathBuf>,
    pub max_repairs: u8,
    pub max_planner_calls: u8,
    pub max_executor_calls: u8,
    pub max_tool_calls: usize,
    pub max_model_tokens: u64,
    pub max_context_tokens: u64,
    pub max_read_bytes: usize,
    pub max_tool_output_bytes: usize,
    pub redact_env: Vec<String>,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            sessions_dir: None,
            max_repairs: 2,
            max_planner_calls: 10,
            max_executor_calls: 40,
            max_tool_calls: 80,
            max_model_tokens: 500_000,
            max_context_tokens: 200_000,
            max_read_bytes: 256 * 1024,
            max_tool_output_bytes: 24 * 1024,
            redact_env: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DockerConfig {
    pub runtime: String,
    pub memory_mb: u64,
    pub cpus: String,
    pub pids_limit: u64,
    pub output_limit_bytes: usize,
    pub tmpfs_mb: u64,
}

impl Default for DockerConfig {
    fn default() -> Self {
        Self {
            runtime: "docker".to_owned(),
            memory_mb: 2048,
            cpus: "2".to_owned(),
            pids_limit: 256,
            output_limit_bytes: 1024 * 1024,
            tmpfs_mb: 256,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RolesConfig {
    pub planner: RoleConfig,
    pub executor: RoleConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RoleConfig {
    pub provider: ProviderKind,
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f32 {
    0.2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Ollama,
    Cohere,
    Openai,
    Anthropic,
    Compatible,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Cohere => "cohere",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Compatible => "compatible",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderSettings {
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub timeout_secs: Option<u64>,
    pub trust_custom_endpoint: bool,
    /// Whether the configured model supports tool calling. Both roles need it,
    /// so this exists to fail early and honestly rather than silently sending
    /// an unsupported request.
    pub tools: bool,
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            base_url: None,
            api_key_env: None,
            timeout_secs: None,
            trust_custom_endpoint: false,
            tools: true,
        }
    }
}

impl ProviderSettings {
    pub fn tools_enabled(&self) -> bool {
        self.tools
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProvidersConfig {
    pub ollama: Option<ProviderSettings>,
    pub cohere: Option<ProviderSettings>,
    pub openai: Option<ProviderSettings>,
    pub anthropic: Option<ProviderSettings>,
    pub compatible: Option<ProviderSettings>,
}

impl ProvidersConfig {
    pub fn get(&self, kind: ProviderKind) -> Option<&ProviderSettings> {
        match kind {
            ProviderKind::Ollama => self.ollama.as_ref(),
            ProviderKind::Cohere => self.cohere.as_ref(),
            ProviderKind::Openai => self.openai.as_ref(),
            ProviderKind::Anthropic => self.anthropic.as_ref(),
            ProviderKind::Compatible => self.compatible.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TestProfile {
    pub name: String,
    pub image: String,
    pub command: Vec<String>,
    #[serde(default = "default_cwd")]
    pub cwd: String,
    #[serde(default = "default_test_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub allow_network: bool,
    #[serde(default)]
    pub prepare_command: Vec<String>,
    #[serde(default)]
    pub prepare_allow_network: bool,
}

fn default_cwd() -> String {
    ".".to_owned()
}

fn default_test_timeout() -> u64 {
    600
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        let config: Self = toml::from_str(&text)
            .with_context(|| format!("failed to parse config {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.harness.max_repairs > 5 {
            bail!("harness.max_repairs must be 5 or less");
        }
        if self.harness.max_planner_calls == 0 || self.harness.max_planner_calls > 50 {
            bail!("harness.max_planner_calls must be between 1 and 50");
        }
        if self.harness.max_executor_calls == 0 || self.harness.max_executor_calls > 200 {
            bail!("harness.max_executor_calls must be between 1 and 200");
        }
        if self.harness.max_tool_calls == 0 {
            bail!("harness.max_tool_calls must be greater than zero");
        }
        if self.harness.max_read_bytes < 1024 {
            bail!("harness.max_read_bytes must be at least 1024");
        }
        if self.harness.max_tool_output_bytes < 1024 {
            bail!("harness.max_tool_output_bytes must be at least 1024");
        }
        if self.harness.max_model_tokens == 0 {
            bail!("harness.max_model_tokens must be greater than zero");
        }
        if self.harness.max_context_tokens == 0 {
            bail!("harness.max_context_tokens must be greater than zero");
        }
        if self.harness.max_read_bytes > 8 * 1024 * 1024 {
            bail!("harness.max_read_bytes must be 8 MiB or less");
        }
        if self.harness.max_tool_output_bytes > 4 * 1024 * 1024 {
            bail!("harness.max_tool_output_bytes must be 4 MiB or less");
        }
        for name in &self.harness.redact_env {
            validate_env_name("harness.redact_env", name)?;
        }
        self.docker.validate()?;
        validate_role("roles.planner", &self.roles.planner, self)?;
        validate_role("roles.executor", &self.roles.executor, self)?;

        for (kind, label, settings) in [
            (
                ProviderKind::Ollama,
                "providers.ollama",
                self.providers.ollama.as_ref(),
            ),
            (
                ProviderKind::Cohere,
                "providers.cohere",
                self.providers.cohere.as_ref(),
            ),
            (
                ProviderKind::Openai,
                "providers.openai",
                self.providers.openai.as_ref(),
            ),
            (
                ProviderKind::Anthropic,
                "providers.anthropic",
                self.providers.anthropic.as_ref(),
            ),
            (
                ProviderKind::Compatible,
                "providers.compatible",
                self.providers.compatible.as_ref(),
            ),
        ] {
            if let Some(settings) = settings {
                settings.validate(label)?;
                if kind == ProviderKind::Compatible && settings.base_url.is_none() {
                    bail!("{label}.base_url is required for a compatible endpoint");
                }
                if settings.base_url.is_some()
                    && settings.api_key_env.is_some()
                    && !settings.trust_custom_endpoint
                    && !is_default_endpoint(kind, settings.base_url.as_deref().unwrap_or(""))
                {
                    bail!(
                        "{label} sends credentials to a custom endpoint but trust_custom_endpoint is not set"
                    );
                }
            }
        }

        let mut names = BTreeSet::new();
        for test in &self.tests {
            test.validate()?;
            if !names.insert(test.name.as_str()) {
                bail!("duplicate test profile name {:?}", test.name);
            }
        }
        Ok(())
    }

    /// Private location for session state. There is deliberately no fallback to
    /// the current directory: sessions must never be written into the project
    /// tree where they could leak source or be committed.
    pub fn sessions_dir(&self) -> Result<PathBuf> {
        if let Some(path) = &self.harness.sessions_dir {
            if !path.is_absolute() {
                bail!("harness.sessions_dir must be an absolute path");
            }
            return Ok(path.clone());
        }
        let base = dirs::state_dir()
            .or_else(dirs::data_local_dir)
            .context(
                "could not determine a private state directory; set harness.sessions_dir to an absolute path outside the project",
            )?;
        Ok(base.join("relay"))
    }

    pub fn provider_settings(&self, kind: ProviderKind) -> Result<&ProviderSettings> {
        self.providers
            .get(kind)
            .with_context(|| format!("provider {:?} is used by a role but not configured", kind))
    }

    pub fn secret_values(&self) -> Vec<String> {
        let mut env_names: Vec<&str> = self.harness.redact_env.iter().map(String::as_str).collect();
        for kind in [
            ProviderKind::Ollama,
            ProviderKind::Cohere,
            ProviderKind::Openai,
            ProviderKind::Anthropic,
            ProviderKind::Compatible,
        ] {
            if let Some(settings) = self.providers.get(kind)
                && let Some(name) = &settings.api_key_env
            {
                env_names.push(name);
            }
        }
        env_names
            .into_iter()
            .filter_map(|name| env::var(name).ok())
            .filter(|value| !value.is_empty())
            .collect()
    }
}

impl DockerConfig {
    fn validate(&self) -> Result<()> {
        const RUNTIMES: &[&str] = &["docker", "podman", "nerdctl"];
        if !RUNTIMES.contains(&self.runtime.as_str()) {
            bail!("docker.runtime must be one of {}", RUNTIMES.join(", "));
        }
        if self.memory_mb < 128 || self.memory_mb > 64 * 1024 {
            bail!("docker.memory_mb must be between 128 and 65536");
        }
        let cpus: f64 = self
            .cpus
            .parse()
            .with_context(|| "docker.cpus must be a decimal number")?;
        if !cpus.is_finite() || cpus <= 0.0 || cpus > 64.0 {
            bail!("docker.cpus must be greater than 0 and at most 64");
        }
        if self.pids_limit < 16 || self.pids_limit > 65_536 {
            bail!("docker.pids_limit must be between 16 and 65536");
        }
        if !(4096..=16 * 1024 * 1024).contains(&self.output_limit_bytes) {
            bail!("docker.output_limit_bytes must be between 4096 and 16777216");
        }
        if self.tmpfs_mb < 16 || self.tmpfs_mb > 8192 {
            bail!("docker.tmpfs_mb must be between 16 and 8192");
        }
        Ok(())
    }
}

fn validate_env_name(label: &str, name: &str) -> Result<()> {
    if name.trim().is_empty()
        || name.contains('=')
        || name.contains(char::is_whitespace)
        || name.contains('\0')
    {
        bail!("{label} must be an environment variable name");
    }
    Ok(())
}

pub fn is_default_endpoint(kind: ProviderKind, base_url: &str) -> bool {
    let normalized = base_url.trim_end_matches('/');
    match kind {
        ProviderKind::Ollama => normalized == "http://localhost:11434",
        ProviderKind::Cohere => normalized == "https://api.cohere.com",
        ProviderKind::Openai => normalized == "https://api.openai.com/v1",
        ProviderKind::Anthropic => normalized == "https://api.anthropic.com",
        ProviderKind::Compatible => false,
    }
}

impl ProviderSettings {
    fn validate(&self, label: &str) -> Result<()> {
        if let Some(name) = &self.api_key_env {
            validate_env_name(&format!("{label}.api_key_env"), name)?;
        }
        if let Some(timeout) = self.timeout_secs
            && !(1..=3600).contains(&timeout)
        {
            bail!("{label}.timeout_secs must be between 1 and 3600");
        }
        if let Some(base_url) = &self.base_url {
            validate_url(label, base_url)?;
        }
        Ok(())
    }
}

fn validate_url(label: &str, value: &str) -> Result<()> {
    let url = reqwest::Url::parse(value).with_context(|| format!("{label}.base_url is invalid"))?;
    if url.scheme() != "https" && url.scheme() != "http" {
        bail!("{label}.base_url must use http or https");
    }
    let local = matches!(
        url.host_str(),
        Some("localhost" | "127.0.0.1" | "::1" | "[::1]")
    );
    if url.scheme() == "http" && !local {
        bail!("{label}.base_url must use https unless it targets loopback");
    }
    if url.username() != "" || url.password().is_some() {
        bail!("{label}.base_url must not contain credentials");
    }
    Ok(())
}

fn validate_role(label: &str, role: &RoleConfig, config: &Config) -> Result<()> {
    if role.model.trim().is_empty() {
        bail!("{label}.model must not be empty");
    }
    if role.max_tokens == 0 || role.max_tokens > 65_536 {
        bail!("{label}.max_tokens must be between 1 and 65536");
    }
    if !(0.0..=2.0).contains(&role.temperature) || role.temperature.is_nan() {
        bail!("{label}.temperature must be between 0 and 2");
    }
    let settings = config.providers.get(role.provider).with_context(|| {
        format!(
            "{label}.provider {:?} has no matching entry under providers",
            role.provider
        )
    })?;
    let hosted = matches!(
        role.provider,
        ProviderKind::Cohere | ProviderKind::Openai | ProviderKind::Anthropic
    );
    if hosted && settings.api_key_env.is_none() {
        bail!(
            "{label} uses hosted provider {} but its api_key_env is not configured",
            role.provider.as_str()
        );
    }
    if !settings.tools_enabled() {
        bail!(
            "{label} requires tool calling, but providers.{}.tools is false",
            role.provider.as_str()
        );
    }
    if role.model.len() > 200 || role.model.contains(char::is_whitespace) {
        bail!("{label}.model must be a single token of at most 200 characters");
    }
    Ok(())
}

impl TestProfile {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("test profile name must not be empty");
        }
        if self.image.trim().is_empty() {
            bail!("test profile {:?} image must not be empty", self.name);
        }
        validate_image_reference(&self.name, &self.image)?;
        validate_argv(
            &format!("test profile {:?} command", self.name),
            &self.command,
            true,
        )?;
        validate_relative_path(&format!("test profile {:?} cwd", self.name), &self.cwd)?;
        if !(1..=3600).contains(&self.timeout_secs) {
            bail!(
                "test profile {:?} timeout_secs must be between 1 and 3600",
                self.name
            );
        }
        validate_argv(
            &format!("test profile {:?} prepare_command", self.name),
            &self.prepare_command,
            false,
        )?;
        if self.prepare_allow_network && self.prepare_command.is_empty() {
            bail!(
                "test profile {:?} allows prepare network access without a prepare command",
                self.name
            );
        }
        Ok(())
    }
}

fn validate_image_reference(profile: &str, image: &str) -> Result<()> {
    if image.len() > 255
        || image.starts_with('-')
        || image.starts_with('/')
        || image.contains(char::is_whitespace)
        || image.contains('\0')
    {
        bail!("test profile {profile:?} image is invalid");
    }
    let valid = image.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-' | '/' | ':' | '@')
    });
    if !valid {
        bail!("test profile {profile:?} image contains unsupported characters");
    }
    Ok(())
}

fn validate_argv(label: &str, argv: &[String], required: bool) -> Result<()> {
    if required && argv.is_empty() {
        bail!("{label} must not be empty");
    }
    if argv.len() > 64 {
        bail!("{label} must contain at most 64 arguments");
    }
    for (index, argument) in argv.iter().enumerate() {
        if argument.len() > 4096 || argument.contains('\0') {
            bail!("{label} argument {index} is invalid");
        }
        if index == 0 && argument.trim().is_empty() {
            bail!("{label} executable must not be empty");
        }
    }
    Ok(())
}

pub fn validate_relative_path(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        bail!("{label} must not be empty");
    }
    let path = Path::new(value);
    if path.is_absolute() {
        bail!("{label} must be relative");
    }
    for component in path.components() {
        match component {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => bail!("{label} must not contain '..'"),
            Component::RootDir | Component::Prefix(_) => bail!("{label} must be relative"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ProviderSettings {
        ProviderSettings {
            api_key_env: Some("TEST_KEY".to_owned()),
            ..ProviderSettings::default()
        }
    }

    fn valid_config() -> Config {
        Config {
            harness: HarnessConfig::default(),
            docker: DockerConfig::default(),
            roles: RolesConfig {
                planner: RoleConfig {
                    provider: ProviderKind::Ollama,
                    model: "local".to_owned(),
                    max_tokens: 1000,
                    temperature: 0.1,
                },
                executor: RoleConfig {
                    provider: ProviderKind::Openai,
                    model: "model".to_owned(),
                    max_tokens: 1000,
                    temperature: 0.1,
                },
            },
            providers: ProvidersConfig {
                ollama: Some(ProviderSettings {
                    base_url: Some("http://localhost:11434".to_owned()),
                    ..ProviderSettings::default()
                }),
                openai: Some(settings()),
                ..ProvidersConfig::default()
            },
            tests: vec![],
        }
    }

    #[test]
    fn accepts_a_valid_configuration() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn accepts_loopback_http_but_rejects_remote_http() {
        let mut config = valid_config();
        config.providers.ollama.as_mut().unwrap().base_url = Some("http://example.com".to_owned());
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_credentials_in_urls() {
        let mut config = valid_config();
        config.providers.ollama.as_mut().unwrap().base_url =
            Some("http://user:pass@localhost:11434".to_owned());
        assert!(config.validate().is_err());
    }

    #[test]
    fn requires_hosted_api_key_environment_name() {
        let mut config = valid_config();
        config.providers.openai.as_mut().unwrap().api_key_env = None;
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_traversing_test_cwd() {
        let mut config = valid_config();
        config.tests.push(TestProfile {
            name: "unit".to_owned(),
            image: "image".to_owned(),
            command: vec!["true".to_owned()],
            cwd: "../outside".to_owned(),
            timeout_secs: 10,
            allow_network: false,
            prepare_command: vec![],
            prepare_allow_network: false,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_duplicate_test_names() {
        let mut config = valid_config();
        let profile = TestProfile {
            name: "same".to_owned(),
            image: "image".to_owned(),
            command: vec!["true".to_owned()],
            cwd: ".".to_owned(),
            timeout_secs: 10,
            allow_network: false,
            prepare_command: vec![],
            prepare_allow_network: false,
        };
        config.tests = vec![profile.clone(), profile];
        assert!(config.validate().is_err());
    }
}
