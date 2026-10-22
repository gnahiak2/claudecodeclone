//! Deterministic providers for tests. They never touch the network and make
//! end-to-end coverage possible without hosted credentials.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;

use super::{
    Capabilities, CompletionRequest, CompletionResponse, ModelProvider, ProviderError, Usage,
};

#[derive(Debug)]
pub struct ScriptedProvider {
    responses: Mutex<VecDeque<CompletionResponse>>,
    requests: Mutex<Vec<CompletionRequest>>,
    tools: bool,
}

impl ScriptedProvider {
    pub fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
            tools: true,
        }
    }

    pub fn without_tools() -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
            tools: false,
        }
    }

    pub fn push(&self, response: CompletionResponse) {
        self.responses
            .lock()
            .expect("scripted provider lock")
            .push_back(response);
    }

    pub fn remaining(&self) -> usize {
        self.responses.lock().expect("scripted provider lock").len()
    }

    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.requests
            .lock()
            .expect("scripted provider lock")
            .clone()
    }
}

#[async_trait]
impl ModelProvider for ScriptedProvider {
    fn name(&self) -> &'static str {
        "scripted"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities { tools: self.tools }
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        self.requests
            .lock()
            .expect("scripted provider lock")
            .push(request);
        let response = self
            .responses
            .lock()
            .expect("scripted provider lock")
            .pop_front();
        response.ok_or_else(|| {
            ProviderError::InvalidResponse("scripted provider has no responses left".to_owned())
        })
    }
}

pub fn text_response(text: impl Into<String>) -> CompletionResponse {
    CompletionResponse {
        text: Some(text.into()),
        tool_calls: Vec::new(),
        usage: Usage {
            input_tokens: Some(100),
            output_tokens: Some(50),
        },
    }
}

pub fn tool_response(name: impl Into<String>, arguments: serde_json::Value) -> CompletionResponse {
    CompletionResponse {
        text: None,
        tool_calls: vec![super::ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            name: name.into(),
            arguments,
        }],
        usage: Usage {
            input_tokens: Some(100),
            output_tokens: Some(20),
        },
    }
}
