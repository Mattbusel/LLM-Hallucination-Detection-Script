//! Fetch a completion with token logprobs from an OpenAI-compatible API.
//!
//! Configuration comes from the environment:
//! - `OPENAI_API_KEY` (required)
//! - `OPENAI_BASE_URL` (optional, default `https://api.openai.com/v1`); any
//!   server that implements Chat Completions with `logprobs` works, for
//!   example Hugging Face's router at `https://router.huggingface.co/v1`.

use anyhow::{Context, Result};
use serde_json::json;

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// Build the request body. Temperature 0 so reruns are comparable.
pub fn request_body(prompt: &str, model: &str, max_tokens: u32) -> serde_json::Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": true,
        "top_logprobs": 3
    })
}

/// Send the prompt and return the raw JSON response text.
pub fn fetch(prompt: &str, model: &str, max_tokens: u32) -> Result<String> {
    let key = std::env::var("OPENAI_API_KEY")
        .context("live mode needs OPENAI_API_KEY in the environment")?;
    let base = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let url = format!("{}/chat/completions", base.trim_end_matches('/'));

    let resp = ureq::post(&url)
        .set("Authorization", &format!("Bearer {}", key))
        .set("Content-Type", "application/json")
        .send_string(&request_body(prompt, model, max_tokens).to_string());

    match resp {
        Ok(r) => r.into_string().context("could not read the API response"),
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            anyhow::bail!("{} returned HTTP {}: {}", url, code, body.trim())
        }
        Err(e) => Err(anyhow::anyhow!("request to {} failed: {}", url, e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_asks_for_logprobs() {
        let b = request_body("hi", "m", 50);
        assert_eq!(b["logprobs"], true);
        assert_eq!(b["top_logprobs"], 3);
        assert_eq!(b["messages"][0]["content"], "hi");
        assert_eq!(b["model"], "m");
    }
}
