use async_trait::async_trait;
use serde_json::{Value, json};

use super::http::{headers, post_json};
use super::{
    Capabilities, CompletionRequest, CompletionResponse, Message, ModelProvider, ProviderError,
    ToolCall, Usage, generated_call_id, parse_arguments,
};

pub(super) struct CohereProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    tools: bool,
}

impl CohereProvider {
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
impl ModelProvider for CohereProvider {
    fn name(&self) -> &'static str {
        "cohere"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { tools: self.tools }
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let authorization = format!("Bearer {}", self.api_key);
        let request_headers = headers(&[
            ("content-type", "application/json"),
            ("authorization", &authorization),
            ("x-client-name", "relay-code"),
        ])?;
        let body = request_body(&request);
        let response = post_json(
            &self.client,
            &format!("{}/v2/chat", self.base_url),
            request_headers,
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
                if calls.is_empty() {
                    messages.push(json!({
                        "role": "assistant",
                        "content": content,
                    }));
                } else {
                    let mut message = json!({
                        "role": "assistant",
                        "tool_calls": calls,
                    });
                    if let Some(content) = content.as_deref().filter(|text| !text.is_empty()) {
                        message["tool_plan"] = json!(content);
                    }
                    messages.push(message);
                }
            }
            Message::Tool {
                call_id, content, ..
            } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": [{
                        "type": "document",
                        "document": {"data": content}
                    }]
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
    }
    body
}

fn parse_response(response: &Value) -> Result<CompletionResponse, ProviderError> {
    let message = response.get("message").ok_or_else(|| {
        ProviderError::InvalidResponse("response did not contain a message".to_owned())
    })?;
    let mut text = String::new();
    if let Some(parts) = message.get("content").and_then(Value::as_array) {
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("text")
                && let Some(part) = part.get("text").and_then(Value::as_str)
            {
                text.push_str(part);
            }
        }
    }
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
            .pointer("/usage/tokens/input_tokens")
            .and_then(Value::as_u64),
        output_tokens: response
            .pointer("/usage/tokens/output_tokens")
            .and_then(Value::as_u64),
    };
    super::reject_incomplete(
        response.get("finish_reason").and_then(Value::as_str),
        "Cohere response",
    )?;
    super::validate_tool_calls(&tool_calls)?;
    let text = if text.is_empty() { None } else { Some(text) };
    if text.is_none() && tool_calls.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_cohere_tool_result_documents() {
        let request = CompletionRequest {
            model: "command".to_owned(),
            system: "system".to_owned(),
            messages: vec![
                Message::User {
                    content: "task".to_owned(),
                },
                Message::Assistant {
                    content: Some("plan".to_owned()),
                    tool_calls: vec![ToolCall {
                        id: "c1".to_owned(),
                        name: "read_file".to_owned(),
                        arguments: json!({"path": "x"}),
                    }],
                },
                Message::Tool {
                    call_id: "c1".to_owned(),
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
        assert_eq!(body["messages"][2]["tool_plan"], "plan");
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"x\"}"
        );
        assert_eq!(body["messages"][3]["content"][0]["type"], "document");
        assert_eq!(
            body["messages"][3]["content"][0]["document"]["data"],
            "data"
        );
    }

    #[test]
    fn parses_cohere_text_and_tool_calls() {
        let response = json!({
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Looking."}],
                "tool_calls": [{
                    "id": "read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}"
                    }
                }]
            },
            "usage": {"tokens": {"input_tokens": 9, "output_tokens": 3}}
        });
        let parsed = parse_response(&response).unwrap();
        assert_eq!(parsed.text.as_deref(), Some("Looking."));
        assert_eq!(parsed.tool_calls[0].name, "read_file");
        assert_eq!(parsed.tool_calls[0].arguments["path"], "README.md");
        assert_eq!(parsed.usage.input_tokens, Some(9));
    }
}
