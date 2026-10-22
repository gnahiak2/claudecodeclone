use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::providers::ProviderError;
use crate::util::truncate_bytes;

const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const SENSITIVE_HEADERS: &[&str] = &["authorization", "x-api-key", "api-key", "x-goog-api-key"];

pub(super) fn headers(entries: &[(&str, &str)]) -> Result<HeaderMap, ProviderError> {
    let mut headers = HeaderMap::new();
    for (name, value) in entries {
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            ProviderError::Configuration(format!("invalid header name: {error}"))
        })?;
        let mut value = HeaderValue::from_str(value).map_err(|_| {
            ProviderError::Configuration(format!("invalid value for header {name}"))
        })?;
        if SENSITIVE_HEADERS.contains(&name.as_str()) {
            value.set_sensitive(true);
        }
        headers.insert(name, value);
    }
    Ok(headers)
}

pub(super) async fn post_json(
    client: &reqwest::Client,
    url: &str,
    headers: HeaderMap,
    body: &Value,
) -> Result<Value, ProviderError> {
    let mut last_error = None;
    for attempt in 0..3u32 {
        let response = client
            .post(url)
            .headers(headers.clone())
            .json(body)
            .send()
            .await;
        match response {
            Ok(response) => {
                let status = response.status();
                if let Some(length) = response
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok())
                    && length > MAX_RESPONSE_BYTES as u64
                {
                    return Err(ProviderError::InvalidResponse(format!(
                        "provider response is too large ({length} bytes)"
                    )));
                }
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok());
                let text = read_body_limited(response).await?;
                if status.is_success() {
                    return serde_json::from_str(&text).map_err(|error| {
                        ProviderError::InvalidResponse(format!(
                            "response was not valid JSON: {error}; body: {}",
                            truncate_bytes(&text, 1000)
                        ))
                    });
                }
                let retryable = status.as_u16() == 429 || status.as_u16() == 408;
                last_error = Some(ProviderError::Http {
                    status: status.as_u16(),
                    body: truncate_bytes(&text, 2000),
                });
                if !retryable || attempt == 2 {
                    break;
                }
                let delay = retry_after
                    .map(|seconds| seconds.clamp(1, 10))
                    .unwrap_or(1 << attempt);
                tokio::time::sleep(Duration::from_secs(delay)).await;
            }
            Err(error) => {
                let retryable = error.is_timeout() || error.is_connect();
                last_error = Some(ProviderError::Transport(error));
                if !retryable || attempt == 2 {
                    break;
                }
                tokio::time::sleep(Duration::from_secs(1 << attempt)).await;
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        ProviderError::InvalidResponse("request failed without an error".to_owned())
    }))
}

async fn read_body_limited(mut response: reqwest::Response) -> Result<String, ProviderError> {
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(ProviderError::InvalidResponse(format!(
                "provider response exceeded the {} MiB limit",
                MAX_RESPONSE_BYTES / (1024 * 1024)
            )));
        }
        body.extend_from_slice(&chunk);
    }
    String::from_utf8(body)
        .map_err(|_| ProviderError::InvalidResponse("response was not valid UTF-8".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_header_injection() {
        assert!(headers(&[("x-test", "value\r\ninjected: yes")]).is_err());
    }

    #[test]
    fn marks_credential_headers_sensitive() {
        let headers = headers(&[("authorization", "Bearer secret")]).unwrap();
        assert!(headers.get("authorization").unwrap().is_sensitive());
    }
}
