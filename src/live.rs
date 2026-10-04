//! Fetch a completion with token logprobs from an OpenAI-compatible API.
//!
//! Pick a provider preset with [`Provider`]; each one knows its default base
//! URL, the environment variable that holds its API key (if it needs one), a
//! default model where one is stable enough to hardcode, and how that API
//! wants logprobs requested.
//!
//! Without a preset the behaviour is the same as before 0.4.0: OpenAI, with
//! `OPENAI_API_KEY` (required) and `OPENAI_BASE_URL` (optional). Any server
//! that implements Chat Completions with `logprobs` works through `base_url`.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

/// OpenAI's API base URL.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Model used with the OpenAI preset when none is given.
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// Seconds before a `--live` request is abandoned.
pub const REQUEST_TIMEOUT_SECS: u64 = 120;

/// Number of alternatives requested per token.
pub const TOP_LOGPROBS: u32 = 3;

/// How a provider wants logprobs requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogprobsParam {
    /// OpenAI style: `"logprobs": true, "top_logprobs": N`.
    BoolWithTop,
    /// Together style: `"logprobs": N` (an integer, the number of alternatives).
    Integer,
}

/// A provider preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// OpenAI (`OPENAI_API_KEY`).
    OpenAi,
    /// OpenRouter (`OPENROUTER_API_KEY`).
    OpenRouter,
    /// Together AI (`TOGETHER_API_KEY`).
    Together,
    /// A vLLM server (`VLLM_API_KEY` if it was started with one).
    Vllm,
    /// A local Ollama (no key).
    Ollama,
}

impl Provider {
    /// Every preset, in `--provider` order.
    pub const ALL: [Provider; 5] = [
        Provider::OpenAi,
        Provider::OpenRouter,
        Provider::Together,
        Provider::Vllm,
        Provider::Ollama,
    ];

    /// The `--provider` name.
    pub fn name(self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::OpenRouter => "openrouter",
            Provider::Together => "together",
            Provider::Vllm => "vllm",
            Provider::Ollama => "ollama",
        }
    }

    /// Preset for a `--provider` name (case-insensitive).
    pub fn from_name(name: &str) -> Option<Provider> {
        let n = name.trim().to_ascii_lowercase();
        Provider::ALL.into_iter().find(|p| p.name() == n)
    }

    /// Base URL used when `--base-url` is not given.
    pub fn default_base_url(self) -> &'static str {
        match self {
            Provider::OpenAi => DEFAULT_BASE_URL,
            Provider::OpenRouter => "https://openrouter.ai/api/v1",
            Provider::Together => "https://api.together.ai/v1",
            Provider::Vllm => "http://localhost:8000/v1",
            Provider::Ollama => "http://localhost:11434/v1",
        }
    }

    /// Environment variable holding the API key, and whether it is required.
    /// vLLM only needs a key when the server was started with `--api-key`.
    pub fn key_env(self) -> Option<(&'static str, bool)> {
        match self {
            Provider::OpenAi => Some(("OPENAI_API_KEY", true)),
            Provider::OpenRouter => Some(("OPENROUTER_API_KEY", true)),
            Provider::Together => Some(("TOGETHER_API_KEY", true)),
            Provider::Vllm => Some(("VLLM_API_KEY", false)),
            Provider::Ollama => None,
        }
    }

    /// Model used when `--model` is not given. `None` means the user must
    /// pick one, because the provider's catalog (or the local server) decides.
    pub fn default_model(self) -> Option<&'static str> {
        match self {
            Provider::OpenAi => Some(DEFAULT_MODEL),
            Provider::OpenRouter => Some("openai/gpt-4o-mini"),
            Provider::Together | Provider::Vllm | Provider::Ollama => None,
        }
    }

    /// What to suggest when no model was given.
    pub fn model_hint(self) -> &'static str {
        match self {
            Provider::Together => {
                "a Together model id (GET https://api.together.ai/v1/models lists them)"
            }
            Provider::Vllm => "the model the server was started with (see GET /v1/models)",
            Provider::Ollama => {
                "a model you have pulled, as listed by `ollama list`, e.g. qwen2.5-coder:14b"
            }
            _ => "a model id",
        }
    }

    /// How this provider wants logprobs requested.
    pub fn logprobs_param(self) -> LogprobsParam {
        match self {
            Provider::Together => LogprobsParam::Integer,
            _ => LogprobsParam::BoolWithTop,
        }
    }
}

/// Resolved settings for one request.
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    /// Provider preset.
    pub provider: Provider,
    /// Base URL, the part before `/chat/completions`.
    pub base_url: String,
    /// Model id.
    pub model: String,
    /// API key, if the provider uses one.
    pub api_key: Option<String>,
}

impl Target {
    /// Resolve a provider, optional overrides and an environment lookup into a
    /// request target. `env` is a parameter so this can be tested without
    /// touching the process environment.
    ///
    /// For the OpenAI preset, `OPENAI_BASE_URL` still applies (backwards
    /// compatible with 0.3); `base_url` beats it.
    pub fn resolve(
        provider: Provider,
        base_url: Option<&str>,
        model: Option<&str>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Target> {
        let base_url = match base_url {
            Some(u) => u.to_string(),
            None if provider == Provider::OpenAi => {
                env("OPENAI_BASE_URL").unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
            }
            None => provider.default_base_url().to_string(),
        };
        let base_url = base_url.trim().trim_end_matches('/').to_string();
        if base_url.is_empty() {
            bail!("--base-url is empty");
        }

        let model = match model.or(provider.default_model()) {
            Some(m) => m.to_string(),
            None => bail!(
                "--provider {} needs --model: {}",
                provider.name(),
                provider.model_hint()
            ),
        };

        let api_key = match provider.key_env() {
            Some((var, required)) => match env(var).filter(|k| !k.trim().is_empty()) {
                Some(k) => Some(k),
                None if required => bail!(
                    "--provider {} needs {} in the environment",
                    provider.name(),
                    var
                ),
                None => None,
            },
            None => None,
        };

        Ok(Target {
            provider,
            base_url,
            model,
            api_key,
        })
    }

    /// The Chat Completions URL.
    pub fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// Build the request body. Temperature 0 so reruns are comparable.
    pub fn body(&self, prompt: &str, max_tokens: u32) -> Value {
        let mut b = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        });
        match self.provider.logprobs_param() {
            LogprobsParam::BoolWithTop => {
                b["logprobs"] = json!(true);
                b["top_logprobs"] = json!(TOP_LOGPROBS);
            }
            LogprobsParam::Integer => {
                b["logprobs"] = json!(TOP_LOGPROBS);
            }
        }
        if self.provider == Provider::OpenRouter {
            // Only route to upstream providers that support every parameter
            // in the request, logprobs included.
            b["provider"] = json!({"require_parameters": true});
        }
        b
    }
}

/// Build an OpenAI-style request body (kept for library callers of 0.3).
pub fn request_body(prompt: &str, model: &str, max_tokens: u32) -> Value {
    Target {
        provider: Provider::OpenAi,
        base_url: DEFAULT_BASE_URL.to_string(),
        model: model.to_string(),
        api_key: None,
    }
    .body(prompt, max_tokens)
}

/// Check that a successful response actually carries token logprobs, and
/// explain clearly when it does not (instead of an empty report).
pub fn check_has_logprobs(raw: &str, target: &Target) -> Result<()> {
    let v: Value = serde_json::from_str(raw).context("the API response is not JSON")?;
    if let Some(err) = v.get("error") {
        bail!("{} returned an error: {}", target.url(), err);
    }
    let choice = v.pointer("/choices/0");
    let lp = choice.and_then(|c| c.get("logprobs"));
    let has = match lp {
        Some(Value::Object(o)) => {
            o.get("content")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty())
                || o.get("tokens")
                    .and_then(Value::as_array)
                    .is_some_and(|a| !a.is_empty())
        }
        Some(Value::Array(a)) => !a.is_empty(),
        _ => false,
    };
    if has {
        return Ok(());
    }
    let answer = choice
        .and_then(|c| c.pointer("/message/content"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let mut msg = format!(
        "{} (model {}) answered, but returned no token logprobs, so there is nothing to analyze.",
        target.provider.name(),
        target.model
    );
    msg.push_str(match target.provider {
        Provider::Ollama => {
            " Ollama returns logprobs on its OpenAI-compatible endpoint in recent versions; update Ollama and try again."
        }
        Provider::OpenRouter => {
            " Pick a model whose upstream provider supports logprobs (the model's page on openrouter.ai lists supported parameters)."
        }
        Provider::OpenAi => " Not every model returns logprobs; OpenAI's logprobs examples use gpt-4o-mini.",
        _ => " This model or server may not support logprobs.",
    });
    if !answer.is_empty() {
        let short: String = answer.chars().take(200).collect();
        msg.push_str(&format!("\nThe answer was: {short}"));
    }
    bail!(msg)
}

/// Send the prompt and return the raw JSON response text.
pub fn fetch_target(target: &Target, prompt: &str, max_tokens: u32) -> Result<String> {
    let url = target.url();
    // ureq has no overall timeout by default, so a stalled server would hang
    // the CLI (and a CI job) forever.
    let mut req = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .set("Content-Type", "application/json");
    if let Some(key) = &target.api_key {
        req = req.set("Authorization", &format!("Bearer {}", key));
    }
    if target.provider == Provider::OpenRouter {
        req = req
            .set(
                "HTTP-Referer",
                "https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script",
            )
            .set("X-Title", "llm-token-visualizer");
    }
    let resp = req.send_string(&target.body(prompt, max_tokens).to_string());

    match resp {
        Ok(r) => r.into_string().context("could not read the API response"),
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            bail!("{} returned HTTP {}: {}", url, code, body.trim())
        }
        Err(e) => {
            let local = matches!(target.provider, Provider::Ollama | Provider::Vllm);
            if local {
                bail!(
                    "request to {} failed: {} (is the {} server running?)",
                    url,
                    e,
                    target.provider.name()
                )
            }
            Err(anyhow::anyhow!("request to {} failed: {}", url, e))
        }
    }
}

/// OpenAI with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` (0.3 behaviour).
pub fn fetch(prompt: &str, model: &str, max_tokens: u32) -> Result<String> {
    let t = Target::resolve(Provider::OpenAi, None, Some(model), |k| {
        std::env::var(k).ok()
    })?;
    fetch_target(&t, prompt, max_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn request_asks_for_logprobs() {
        let b = request_body("hi", "m", 50);
        assert_eq!(b["logprobs"], true);
        assert_eq!(b["top_logprobs"], 3);
        assert_eq!(b["messages"][0]["content"], "hi");
        assert_eq!(b["model"], "m");
        assert_eq!(b["temperature"], 0);
        assert!(b.get("provider").is_none());
    }

    #[test]
    fn provider_names_round_trip() {
        for p in Provider::ALL {
            assert_eq!(Provider::from_name(p.name()), Some(p));
        }
        assert_eq!(
            Provider::from_name("OpenRouter"),
            Some(Provider::OpenRouter)
        );
        assert_eq!(Provider::from_name("anthropic"), None);
    }

    #[test]
    fn openai_default_is_backwards_compatible() {
        let t = Target::resolve(
            Provider::OpenAi,
            None,
            None,
            env(&[("OPENAI_API_KEY", "sk")]),
        )
        .unwrap();
        assert_eq!(t.url(), "https://api.openai.com/v1/chat/completions");
        assert_eq!(t.model, "gpt-4o-mini");
        assert_eq!(t.api_key.as_deref(), Some("sk"));

        let t = Target::resolve(
            Provider::OpenAi,
            None,
            None,
            env(&[
                ("OPENAI_API_KEY", "sk"),
                ("OPENAI_BASE_URL", "https://x.test/v1/"),
            ]),
        )
        .unwrap();
        assert_eq!(t.url(), "https://x.test/v1/chat/completions");

        let e = Target::resolve(Provider::OpenAi, None, None, env(&[])).unwrap_err();
        assert!(e.to_string().contains("OPENAI_API_KEY"), "{e}");
    }

    #[test]
    fn base_url_flag_beats_env_and_preset() {
        let t = Target::resolve(
            Provider::OpenAi,
            Some("http://h:1/v1/"),
            None,
            env(&[
                ("OPENAI_API_KEY", "sk"),
                ("OPENAI_BASE_URL", "https://x.test/v1"),
            ]),
        )
        .unwrap();
        assert_eq!(t.url(), "http://h:1/v1/chat/completions");
        let t = Target::resolve(
            Provider::Ollama,
            Some("http://box:11434/v1"),
            Some("m"),
            env(&[]),
        )
        .unwrap();
        assert_eq!(t.url(), "http://box:11434/v1/chat/completions");
    }

    #[test]
    fn openrouter_request() {
        let t = Target::resolve(
            Provider::OpenRouter,
            None,
            None,
            env(&[
                ("OPENROUTER_API_KEY", "or"),
                ("OPENAI_BASE_URL", "https://ignored"),
            ]),
        )
        .unwrap();
        assert_eq!(t.url(), "https://openrouter.ai/api/v1/chat/completions");
        assert_eq!(t.model, "openai/gpt-4o-mini");
        assert_eq!(t.api_key.as_deref(), Some("or"));
        let b = t.body("q", 10);
        assert_eq!(b["logprobs"], true);
        assert_eq!(b["top_logprobs"], 3);
        assert_eq!(b["provider"]["require_parameters"], true);
        let e = Target::resolve(Provider::OpenRouter, None, None, env(&[])).unwrap_err();
        assert!(e.to_string().contains("OPENROUTER_API_KEY"), "{e}");
    }

    #[test]
    fn together_request_uses_integer_logprobs() {
        let t = Target::resolve(
            Provider::Together,
            None,
            Some("some/model"),
            env(&[("TOGETHER_API_KEY", "tg")]),
        )
        .unwrap();
        assert_eq!(t.url(), "https://api.together.ai/v1/chat/completions");
        let b = t.body("q", 10);
        assert_eq!(b["logprobs"], 3);
        assert!(b.get("top_logprobs").is_none());
        let e = Target::resolve(
            Provider::Together,
            None,
            None,
            env(&[("TOGETHER_API_KEY", "tg")]),
        )
        .unwrap_err();
        assert!(e.to_string().contains("--model"), "{e}");
    }

    #[test]
    fn local_providers_need_no_key() {
        let t =
            Target::resolve(Provider::Ollama, None, Some("qwen2.5-coder:14b"), env(&[])).unwrap();
        assert_eq!(t.url(), "http://localhost:11434/v1/chat/completions");
        assert_eq!(t.api_key, None);
        assert_eq!(t.body("q", 5)["logprobs"], true);

        let t = Target::resolve(Provider::Vllm, None, Some("m"), env(&[])).unwrap();
        assert_eq!(t.url(), "http://localhost:8000/v1/chat/completions");
        assert_eq!(t.api_key, None);
        let t = Target::resolve(
            Provider::Vllm,
            None,
            Some("m"),
            env(&[("VLLM_API_KEY", "v")]),
        )
        .unwrap();
        assert_eq!(t.api_key.as_deref(), Some("v"));

        let e = Target::resolve(Provider::Ollama, None, None, env(&[])).unwrap_err();
        assert!(e.to_string().contains("ollama list"), "{e}");
    }

    fn target(p: Provider) -> Target {
        Target {
            provider: p,
            base_url: p.default_base_url().into(),
            model: "m".into(),
            api_key: None,
        }
    }

    #[test]
    fn missing_logprobs_is_explained() {
        let raw = r#"{"choices":[{"message":{"content":"Rembrandt."},"logprobs":null}]}"#;
        let e = check_has_logprobs(raw, &target(Provider::Ollama))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no token logprobs"), "{e}");
        assert!(e.contains("Rembrandt."), "{e}");
        let raw = r#"{"choices":[{"message":{"content":"x"}}]}"#;
        assert!(check_has_logprobs(raw, &target(Provider::OpenRouter)).is_err());
        let raw = r#"{"choices":[{"logprobs":{"content":[]}}]}"#;
        assert!(check_has_logprobs(raw, &target(Provider::OpenAi)).is_err());
    }

    #[test]
    fn present_logprobs_pass_the_check() {
        let raw = r#"{"choices":[{"logprobs":{"content":[{"token":"a","logprob":-0.1}]}}]}"#;
        assert!(check_has_logprobs(raw, &target(Provider::OpenAi)).is_ok());
        let raw = r#"{"choices":[{"logprobs":{"tokens":["a"],"token_logprobs":[-0.1]}}]}"#;
        assert!(check_has_logprobs(raw, &target(Provider::Together)).is_ok());
    }
}
