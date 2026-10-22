use async_trait::async_trait;
use serde_json::{Value, json};

use super::http::{headers, post_json};
use super::{
    Capabilities, CompletionRequest, CompletionResponse, Message, ModelProvider, ProviderError,
    ToolCall, Usage, generated_call_id, parse_arguments,
};

pub(super) struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    tools: bool,
}

impl AnthropicProvider {
    pub(super) fn new(
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        tools: bool,
    ) -> Self {
        Self {
            client,
            base_url,
            api_key,
            tools,
        }
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { tools: self.tools }
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let request_headers = headers(&[
            ("content-type", "application/json"),
            ("x-api-key", &self.api_key),
            ("anthropic-version", "2023-06-01"),
        ])?;
        let body = request_body(&request);
        let response = post_json(
            &self.client,
            &format!("{}/v1/messages", self.base_url),
            request_headers,
            &body,
        )
        .await?;
        parse_response(&response)
    }
}

fn request_body(request: &CompletionRequest) -> Value {
    let mut messages: Vec<Value> = Vec::with_capacity(request.messages.len());
    let mut pending_tool_results: Vec<Value> = Vec::new();

    let flush_results = |messages: &mut Vec<Value>, pending: &mut Vec<Value>| {
        if !pending.is_empty() {
            messages.push(json!({
                "role": "user",
                "content": std::mem::take(pending),
            }));
        }
    };

    for message in &request.messages {
        match message {
            Message::User { content } => {
                flush_results(&mut messages, &mut pending_tool_results);
                messages.push(json!({"role": "user", "content": content}));
            }
            Message::Assistant {
                content,
                tool_calls,
            } => {
                flush_results(&mut messages, &mut pending_tool_results);
                let mut blocks = Vec::new();
                if let Some(content) = content.as_deref().filter(|text| !text.is_empty()) {
                    blocks.push(json!({"type": "text", "text": content}));
                }
                for call in tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }));
                }
                if !blocks.is_empty() {
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            Message::Tool {
                call_id, content, ..
            } => {
                pending_tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                }));
            }
        }
    }
    flush_results(&mut messages, &mut pending_tool_results);

    let tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        })
        .collect();
    let mut body = json!({
        "model": request.model,
        "system": request.system,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": false,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    body
}

fn parse_response(response: &Value) -> Result<CompletionResponse, ProviderError> {
    let blocks = response
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::InvalidResponse("response did not contain a content array".to_owned())
        })?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                }
            }
            Some("tool_use") => {
                let name = block.get("name").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::InvalidResponse(format!("tool_use block {index} has no name"))
                })?;
                let arguments = block
                    .get("input")
                    .map(parse_arguments)
                    .transpose()?
                    .unwrap_or_else(|| json!({}));
                let id = block
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
            _ => {}
        }
    }
    let usage = Usage {
        input_tokens: response
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64),
        output_tokens: response
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64),
    };
    super::reject_incomplete(
        response.get("stop_reason").and_then(Value::as_str),
        "Anthropic response",
    )?;
    super::validate_tool_calls(&tool_calls)?;
    let text = if text.is_empty() { None } else { Some(text) };
    if text.is_none() && tool_calls.is_empty() {
        return Err(ProviderError::InvalidResponse(
            "assistant response contained neither text nor tool_use blocks".to_owned(),
        ));
    }
    Ok(CompletionResponse {
        text,
        tool_calls,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_tool_results_into_user_blocks() {
        let request = CompletionRequest {
            model: "claude".to_owned(),
            system: "system".to_owned(),
            messages: vec![
                Message::User {
                    content: "task".to_owned(),
                },
                Message::Assistant {
                    content: Some("checking".to_owned()),
                    tool_calls: vec![
                        ToolCall {
                            id: "one".to_owned(),
                            name: "read_file".to_owned(),
                            arguments: json!({"path": "a"}),
                        },
                        ToolCall {
                            id: "two".to_owned(),
                            name: "read_file".to_owned(),
                            arguments: json!({"path": "b"}),
                        },
                    ],
                },
                Message::Tool {
                    call_id: "one".to_owned(),
                    name: "read_file".to_owned(),
                    content: "A".to_owned(),
                },
                Message::Tool {
                    call_id: "two".to_owned(),
                    name: "read_file".to_owned(),
                    content: "B".to_owned(),
                },
            ],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.2,
        };
        let body = request_body(&request);
        assert_eq!(body["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(body["messages"][2]["role"], "user");
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(body["messages"][2]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn parses_text_and_tool_use() {
        let response = json!({
            "content": [
                {"type": "text", "text": "I will inspect."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "x"}}
            ],
            "usage": {"input_tokens": 12, "output_tokens": 8},
            "stop_reason": "tool_use"
        });
        let parsed = parse_response(&response).unwrap();
        assert_eq!(parsed.text.as_deref(), Some("I will inspect."));
        assert_eq!(parsed.tool_calls[0].id, "toolu_1");
        assert_eq!(parsed.tool_calls[0].arguments["path"], "x");
        assert_eq!(parsed.usage.output_tokens, Some(8));
    }

    #[test]
    fn rejects_empty_content() {
        assert!(parse_response(&json!({"content": []})).is_err());
    }
}
