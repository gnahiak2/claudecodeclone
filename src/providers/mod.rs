mod anthropic;
mod cohere;
mod http;
mod ollama;
mod openai;
pub mod scripted;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{ProviderKind, ProviderSettings};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
    },
    Tool {
        call_id: String,
        name: String,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = sum(self.input_tokens, other.input_tokens);
        self.output_tokens = sum(self.output_tokens, other.output_tokens);
    }
}

fn sum(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub text: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub tools: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("provider configuration error: {0}")]
    Configuration(String),
    #[error("provider transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("provider returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("provider returned an invalid response: {0}")]
    InvalidResponse(String),
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError>;
}

pub fn create_provider(
    kind: ProviderKind,
    settings: &ProviderSettings,
) -> Result<Arc<dyn ModelProvider>> {
    let timeout = std::time::Duration::from_secs(settings.timeout_secs.unwrap_or(120));
    let api_key =
        match &settings.api_key_env {
            Some(name) => Some(std::env::var(name).with_context(|| {
                format!("environment variable {name} is required but is not set")
            })?),
            None => None,
        };

    let default_base = match kind {
        ProviderKind::Ollama => "http://localhost:11434",
        ProviderKind::Cohere => "https://api.cohere.com",
        ProviderKind::Openai => "https://api.openai.com/v1",
        ProviderKind::Anthropic => "https://api.anthropic.com",
        ProviderKind::Compatible => {
            if settings.base_url.is_none() {
                bail!("providers.compatible.base_url is required");
            }
            ""
        }
    };
    let base_url = settings
        .base_url
        .as_deref()
        .unwrap_or(default_base)
        .trim_end_matches('/')
        .to_owned();

    if api_key.is_some() && settings.base_url.as_deref() != Some(default_base) {
        let trusted = settings.trust_custom_endpoint || base_url == default_base;
        if !trusted {
            bail!(
                "refusing to send credentials to a custom endpoint; set providers.{}.trust_custom_endpoint = true after verifying {}",
                kind.as_str(),
                base_url
            );
        }
    }

    // Credentials must never follow a redirect to an unverified host. TLS
    // certificate verification keeps reqwest's secure default.
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(15))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("relay-code/", env!("CARGO_PKG_VERSION")))
        .build()?;

    let tools = settings.tools_enabled();
    let provider: Arc<dyn ModelProvider> = match kind {
        ProviderKind::Ollama => Arc::new(ollama::OllamaProvider::new(client, base_url, tools)),
        ProviderKind::Cohere => Arc::new(cohere::CohereProvider::new(
            client,
            base_url,
            require_key(kind, api_key)?,
            tools,
        )),
        ProviderKind::Openai => Arc::new(openai::OpenAiProvider::new(
            client,
            base_url,
            Some(require_key(kind, api_key)?),
            "openai",
            tools,
        )),
        ProviderKind::Anthropic => Arc::new(anthropic::AnthropicProvider::new(
            client,
            base_url,
            require_key(kind, api_key)?,
            tools,
        )),
        ProviderKind::Compatible => Arc::new(openai::OpenAiProvider::new(
            client,
            base_url,
            api_key,
            "compatible",
            tools,
        )),
    };
    if !provider.capabilities().tools {
        bail!(
            "provider {} is configured without tool support; relay requires tool calling for both roles",
            kind.as_str()
        );
    }
    Ok(provider)
}

fn require_key(kind: ProviderKind, key: Option<String>) -> Result<String> {
    key.with_context(|| format!("provider {} requires an API key", kind.as_str()))
}

pub(crate) fn parse_arguments(value: &Value) -> Result<Value, ProviderError> {
    let parsed = match value {
        Value::Null => Value::Object(Default::default()),
        Value::Object(_) => value.clone(),
        Value::String(text) => {
            if text.trim().is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(text).map_err(|error| {
                    ProviderError::InvalidResponse(format!(
                        "tool arguments are not valid JSON: {error}"
                    ))
                })?
            }
        }
        _ => {
            return Err(ProviderError::InvalidResponse(
                "tool arguments must be a JSON object or an object encoded as a string".to_owned(),
            ));
        }
    };
    if !parsed.is_object() {
        return Err(ProviderError::InvalidResponse(
            "tool arguments must be a JSON object".to_owned(),
        ));
    }
    Ok(parsed)
}

/// Reject responses that stopped because of a token limit or a safety filter.
/// Tool calls from incomplete responses must never be dispatched.
pub(crate) fn reject_incomplete(reason: Option<&str>, context: &str) -> Result<(), ProviderError> {
    let Some(reason) = reason else {
        return Ok(());
    };
    let normalized = reason.to_ascii_lowercase();
    if matches!(
        normalized.as_str(),
        "length" | "max_tokens" | "content_filter" | "refusal" | "incomplete"
    ) {
        return Err(ProviderError::InvalidResponse(format!(
            "{context} stopped with finish reason {reason:?}; refusing an incomplete response"
        )));
    }
    Ok(())
}

pub(crate) fn validate_tool_calls(calls: &[ToolCall]) -> Result<(), ProviderError> {
    if calls.len() > 64 {
        return Err(ProviderError::InvalidResponse(
            "response contained too many tool calls".to_owned(),
        ));
    }
    for call in calls {
        if call.name.is_empty() || call.name.len() > 128 {
            return Err(ProviderError::InvalidResponse(
                "tool call name is invalid".to_owned(),
            ));
        }
        if !call.name.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        }) {
            return Err(ProviderError::InvalidResponse(format!(
                "tool call name {:?} contains unsupported characters",
                call.name
            )));
        }
        if !call.arguments.is_object() {
            return Err(ProviderError::InvalidResponse(
                "tool call arguments must be an object".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn generated_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_object_and_string_arguments() {
        let object = serde_json::json!({"path": "src/main.rs"});
        assert_eq!(parse_arguments(&object).unwrap(), object);
        let text = serde_json::json!("{\"path\":\"src/main.rs\"}");
        assert_eq!(parse_arguments(&text).unwrap(), object);
    }

    #[test]
    fn rejects_invalid_tool_arguments() {
        let text = serde_json::json!("{not json");
        assert!(parse_arguments(&text).is_err());
    }

    #[test]
    fn usage_addition_handles_missing_values() {
        let mut usage = Usage {
            input_tokens: Some(3),
            output_tokens: None,
        };
        usage.add(&Usage {
            input_tokens: Some(5),
            output_tokens: Some(2),
        });
        assert_eq!(usage.input_tokens, Some(8));
        assert_eq!(usage.output_tokens, Some(2));
    }
}
