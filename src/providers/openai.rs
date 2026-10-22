use async_trait::async_trait;
use serde_json::{Value, json};

use super::http::{headers, post_json};
use super::{
    Capabilities, CompletionRequest, CompletionResponse, Message, ModelProvider, ProviderError,
    ToolCall, Usage, generated_call_id, parse_arguments,
};

pub(super) struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    name: &'static str,
    tools: bool,
}

impl OpenAiProvider {
    pub(super) fn new(
        client: reqwest::Client,
        base_url: String,
        api_key: Option<String>,
        name: &'static str,
        tools: bool,
    ) -> Self {
        Self {
            client,
            base_url,
            api_key,
            name,
            tools,
        }
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    fn name(&self) -> &'static str {
        self.name
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { tools: self.tools }
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let mut header_entries = vec![("content-type", "application/json")];
        let auth;
        if let Some(key) = &self.api_key {
            auth = format!("Bearer {key}");
            header_entries.push(("authorization", &auth));
        }
        let body = request_body(&request);
        let response = post_json(
            &self.client,
            &format!("{}/chat/completions", self.base_url),
            headers(&header_entries)?,
            &body,
        )
        .await?;
        parse_response(&response)
    }
}

fn request_body(request: &CompletionRequest) -> Value {
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    if !request.system.is_empty() {
        messages.push(json!({"role": "system", "content": request.system}));
    }
    for message in &request.messages {
        match message {
            Message::User { content } => {
                messages.push(json!({"role": "user", "content": content}));
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                let calls: Vec<Value> = tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .unwrap_or_else(|_| "{}".to_owned())
                            }
                        })
                    })
                    .collect();
                let mut message = json!({
                    "role": "assistant",
                    "content": content,
                });
                if !calls.is_empty() {
                    message["tool_calls"] = Value::Array(calls);
                }
                messages.push(message);
            }
            Message::Tool {
                call_id, content, ..
            } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }));
            }
        }
    }
    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect();
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": false,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = json!("auto");
    }
    body
}

fn parse_response(response: &Value) -> Result<CompletionResponse, ProviderError> {
    let choice = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| {
            ProviderError::InvalidResponse("response did not contain choices[0]".to_owned())
        })?;
    let message = choice.get("message").ok_or_else(|| {
        ProviderError::InvalidResponse("response did not contain choices[0].message".to_owned())
    })?;
    let text = message.get("content").and_then(extract_content);
    let mut tool_calls = Vec::new();
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for (index, call) in calls.iter().enumerate() {
            let function = call.get("function").ok_or_else(|| {
                ProviderError::InvalidResponse(format!("tool call {index} has no function"))
            })?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProviderError::InvalidResponse(format!("tool call {index} has no name"))
                })?;
            let arguments = function
                .get("arguments")
                .map(parse_arguments)
                .transpose()?
                .unwrap_or_else(|| json!({}));
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(generated_call_id);
            tool_calls.push(ToolCall {
                id,
                name: name.to_owned(),
                arguments,
            });
        }
    }
    let usage = Usage {
        input_tokens: response
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64),
        output_tokens: response
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64),
    };
    super::reject_incomplete(
        choice.get("finish_reason").and_then(Value::as_str),
        "OpenAI response",
    )?;
    super::validate_tool_calls(&tool_calls)?;
    if text.as_deref().is_none_or(str::is_empty) && tool_calls.is_empty() {
        return Err(ProviderError::InvalidResponse(
            "assistant response contained neither text nor tool calls".to_owned(),
        ));
    }
    Ok(CompletionResponse {
        text,
        tool_calls,
        usage,
    })
}

fn extract_content(value: &Value) -> Option<String> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Array(parts) => {
            let text: Vec<&str> = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect();
            if text.is_empty() {
                None
            } else {
                Some(text.join(""))
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_tool_messages() {
        let request = CompletionRequest {
            model: "m".to_owned(),
            system: "system".to_owned(),
            messages: vec![
                Message::User {
                    content: "fix".to_owned(),
                },
                Message::Assistant {
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "call_1".to_owned(),
                        name: "read_file".to_owned(),
                        arguments: json!({"path": "src/main.rs"}),
                    }],
                },
                Message::Tool {
                    call_id: "call_1".to_owned(),
                    name: "read_file".to_owned(),
                    content: "data".to_owned(),
                },
            ],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.2,
        };
        let body = request_body(&request);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][2]["tool_calls"][0]["type"], "function");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"src/main.rs\"}"
        );
        assert_eq!(body["messages"][3]["role"], "tool");
    }

    #[test]
    fn parses_text_and_tool_calls() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": "working",
                    "tool_calls": [{
                        "id": "abc",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\":\"README.md\"}"
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4}
        });
        let parsed = parse_response(&response).unwrap();
        assert_eq!(parsed.text.as_deref(), Some("working"));
        assert_eq!(parsed.tool_calls[0].arguments["path"], "README.md");
        assert_eq!(parsed.usage.input_tokens, Some(10));
    }

    #[test]
    fn parses_content_parts_and_missing_ids() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}],
                    "tool_calls": [{
                        "function": {"name": "list_files", "arguments": {}}
                    }]
                }
            }]
        });
        let parsed = parse_response(&response).unwrap();
        assert_eq!(parsed.text.as_deref(), Some("hello world"));
        assert!(parsed.tool_calls[0].id.starts_with("call_"));
    }

    #[test]
    fn rejects_empty_assistant_responses() {
        let response = json!({"choices": [{"message": {"content": null}}]});
        assert!(parse_response(&response).is_err());
    }
}
