//! Adapters for other crates, each behind its own optional feature.
//!
//! | Feature | Adds |
//! |---|---|
//! | `async-openai` | [`async_openai`]: tokens from an `async_openai` chat completion response |
//!
//! Any other client works through JSON: [`crate::detect::parse_logprobs`]
//! reads OpenAI-style, legacy completions-style and Gemini-style responses.

/// Adapter for [async-openai](https://crates.io/crates/async-openai) (types
/// only, no HTTP client).
#[cfg(feature = "async-openai")]
#[cfg_attr(docsrs, doc(cfg(feature = "async-openai")))]
pub mod async_openai {
    use ::async_openai::types::chat::{ChatCompletionTokenLogprob, CreateChatCompletionResponse};
    use anyhow::{bail, Result};

    use crate::detect::{Alternative, LogprobToken};

    impl From<&ChatCompletionTokenLogprob> for LogprobToken {
        fn from(t: &ChatCompletionTokenLogprob) -> Self {
            let mut top: Vec<Alternative> = t
                .top_logprobs
                .iter()
                .map(|a| Alternative {
                    token: a.token.clone(),
                    logprob: f64::from(a.logprob),
                })
                .collect();
            top.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
            LogprobToken {
                token: t.token.clone(),
                logprob: f64::from(t.logprob),
                top_logprobs: top,
            }
        }
    }

    /// The tokens of the first choice, ready for [`crate::detect::detect`].
    /// Special tokens such as `<|eot_id|>` are dropped, as in
    /// [`crate::detect::parse_logprobs`].
    ///
    /// # Errors
    ///
    /// When the response has no choices or no logprobs (request the
    /// completion with `logprobs(true)` and `top_logprobs(3)`).
    pub fn tokens_from_response(resp: &CreateChatCompletionResponse) -> Result<Vec<LogprobToken>> {
        let Some(choice) = resp.choices.first() else {
            bail!("the response has no choices");
        };
        let Some(content) = choice.logprobs.as_ref().and_then(|l| l.content.as_ref()) else {
            bail!("no token logprobs; request the completion with logprobs(true)");
        };
        let tokens: Vec<LogprobToken> = content
            .iter()
            .map(LogprobToken::from)
            .filter(|t| !(t.token.starts_with("<|") && t.token.ends_with("|>")))
            .collect();
        if tokens.is_empty() {
            bail!("logprobs contain no tokens");
        }
        Ok(tokens)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn same_tokens_as_the_json_parser() {
            let json = include_str!("../examples/logprobs/cuyp.json");
            let resp: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
            let a = tokens_from_response(&resp).unwrap();
            let b = crate::detect::parse_logprobs(json).unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.token, y.token);
                // async-openai stores logprobs as f32.
                assert!((x.logprob - y.logprob).abs() < 1e-6);
            }
            let r = crate::detect::detect(&a, 0.6);
            assert!(r.spans.iter().any(|s| s.text == " Dordrecht"));
        }
    }
}
