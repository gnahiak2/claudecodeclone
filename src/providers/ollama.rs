use async_trait::async_trait;
use serde_json::{Value, json};

use super::http::{headers, post_json};
use super::{
    Capabilities, CompletionRequest, CompletionResponse, Message, ModelProvider, ProviderError,
    ToolCall, Usage, generated_call_id, parse_arguments,
};

pub(super) struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
    tools: bool,
}

impl OllamaProvider {
    pub(super) fn new(client: reqwest::Client, base_url: String, tools: bool) -> Self {
        Self {
            client,
            base_url,
            tools,
        }
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { tools: self.tools }
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let body = request_body(&request);
        let response = post_json(
            &self.client,
            &format!("{}/api/chat", self.base_url),
            headers(&[("content-type", "application/json")])?,
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
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            }
                        })
                    })
                    .collect();
                let mut message = json!({
                    "role": "assistant",
                    "content": content.clone().unwrap_or_default(),
                });
                if !calls.is_empty() {
                    message["tool_calls"] = Value::Array(calls);
                }
                messages.push(message);
            }
            Message::Tool { name, content, .. } => {
                messages.push(json!({
                    "role": "tool",
                    "tool_name": name,
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
        "stream": false,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    body
}

fn parse_response(response: &Value) -> Result<CompletionResponse, ProviderError> {
    if let Some(error) = response.get("error").and_then(Value::as_str) {
        return Err(ProviderError::InvalidResponse(error.to_owned()));
    }
    let message = response.get("message").ok_or_else(|| {
        ProviderError::InvalidResponse("response did not contain a message".to_owned())
    })?;
    let text = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(str::to_owned);
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
            tool_calls.push(ToolCall {
                id: generated_call_id(),
                name: name.to_owned(),
                arguments,
            });
        }
    }
    let usage = Usage {
        input_tokens: response.get("prompt_eval_count").and_then(Value::as_u64),
        output_tokens: response.get("eval_count").and_then(Value::as_u64),
    };
    if response.get("done").and_then(Value::as_bool) == Some(false) {
        return Err(ProviderError::InvalidResponse(
            "Ollama response was not complete".to_owned(),
        ));
    }
    super::reject_incomplete(
        response.get("done_reason").and_then(Value::as_str),
        "Ollama response",
    )?;
    super::validate_tool_calls(&tool_calls)?;
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
    fn builds_tool_messages_with_names() {
        let request = CompletionRequest {
            model: "local".to_owned(),
            system: "system".to_owned(),
            messages: vec![
                Message::Assistant {
                    content: None,
                    tool_calls: vec![ToolCall {
                        id: "x".to_owned(),
                        name: "read_file".to_owned(),
                        arguments: json!({"path": "a"}),
                    }],
                },
                Message::Tool {
                    call_id: "x".to_owned(),
                    name: "read_file".to_owned(),
                    content: "A".to_owned(),
                },
            ],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.1,
        };
        let body = request_body(&request);
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(body["messages"][2]["tool_name"], "read_file");
        assert_eq!(body["options"]["num_predict"], 100);
    }

    #[test]
    fn parses_object_and_string_arguments() {
        let response = json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "one", "arguments": {"path": "a"}}},
                    {"function": {"name": "two", "arguments": "{\"path\":\"b\"}"}}
                ]
            },
            "prompt_eval_count": 7,
            "eval_count": 2
        });
        let parsed = parse_response(&response).unwrap();
        assert!(parsed.text.is_none());
        assert_eq!(parsed.tool_calls[0].arguments["path"], "a");
        assert_eq!(parsed.tool_calls[1].arguments["path"], "b");
        assert_eq!(parsed.usage.input_tokens, Some(7));
    }

    #[test]
    fn propagates_ollama_errors() {
        let response = json!({"error": "model not found"});
        assert!(parse_response(&response).is_err());
    }
}
