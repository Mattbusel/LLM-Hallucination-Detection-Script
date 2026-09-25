//! Reports for detect mode: a confidence heatmap of the answer, the flagged
//! spans, and the alternatives the model weighed at each weak token.
//!
//! Three formats share one color scale ([`heat`]): a colored terminal report,
//! a self-contained HTML page (no scripts, no external assets), and Markdown
//! that reads well in a pull request or issue comment.

use colored::Colorize;
use std::fmt::Write as _;

use crate::detect::{LogprobToken, Report, Span};

/// Where the answer came from, shown in report headers.
#[derive(Debug, Clone, Default)]
pub struct Meta {
    /// e.g. the input file name, or "live".
    pub source: String,
    /// Model name from the response, if it had one.
    pub model: Option<String>,
}

impl Meta {
    /// Read the `model` field from a raw Chat Completions response.
    pub fn from_response(source: impl Into<String>, json: &str) -> Self {
        let model = serde_json::from_str::<serde_json::Value>(json)
            .ok()
            .and_then(|v| v.get("model")?.as_str().map(str::to_string));
        Meta {
            source: source.into(),
            model,
        }
    }
}

/// Map a probability to an RGB color: warm ink for confident tokens, through
/// amber, to vermilion for the least likely ones.
pub fn heat(p: f64) -> (u8, u8, u8) {
    const STOPS: [(f64, (f64, f64, f64)); 4] = [
        (0.25, (232.0, 72.0, 48.0)),
        (0.55, (240.0, 140.0, 52.0)),
        (0.80, (236.0, 192.0, 92.0)),
        (0.97, (214.0, 208.0, 196.0)),
    ];
    let p = p.clamp(0.0, 1.0);
    if p <= STOPS[0].0 {
        let c = STOPS[0].1;
        return (c.0 as u8, c.1 as u8, c.2 as u8);
    }
    for w in STOPS.windows(2) {
        let ((p0, a), (p1, b)) = (w[0], w[1]);
        if p <= p1 {
            let t = (p - p0) / (p1 - p0);
            let mix = |x: f64, y: f64| (x + (y - x) * t).round() as u8;
            return (mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2));
        }
    }
    let c = STOPS[3].1;
    (c.0 as u8, c.1 as u8, c.2 as u8)
}

/// For each token, the 1-based number of the span it belongs to (0 = none).
fn span_index(tokens: &[LogprobToken], report: &Report) -> Vec<usize> {
    let mut idx = vec![0; tokens.len()];
    for (n, s) in report.spans.iter().enumerate() {
        for slot in idx.iter_mut().take(s.end).skip(s.start) {
            *slot = n + 1;
        }
    }
    idx
}

fn summary(report: &Report) -> String {
    match report.spans.len() {
        0 => "No low-confidence spans. That does not prove the answer is correct: \
              models can be confidently wrong."
            .to_string(),
        1 => "1 low-confidence span. Check it before trusting the answer.".to_string(),
        n => format!("{n} low-confidence spans. Check these claims before trusting the answer."),
    }
}

/// The candidates at a span's weakest token, with the chosen token included
/// and marked, most likely first.
fn candidates(span: &Span) -> Vec<(String, f64, bool)> {
    let mut out: Vec<(String, f64, bool)> = span
        .alternatives
        .iter()
        .map(|c| (c.token.clone(), c.prob, c.token == span.weakest_token))
        .collect();
    if !out.iter().any(|c| c.2) {
        out.push((span.weakest_token.clone(), span.min_prob, true));
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn token_range(s: &Span) -> String {
    if s.end - s.start == 1 {
        format!("token {}", s.start)
    } else {
        format!("tokens {}-{}", s.start, s.end - 1)
    }
}

/// Show a token so leading spaces and empty strings stay visible.
fn show(token: &str) -> String {
    format!("{token:?}")
}

// ---------------------------------------------------------------- terminal

/// Confident tokens keep the terminal's own color so the answer stays
/// readable on light and dark themes; only doubt gets a heat color.
fn piece_for(text: &str, p: f64) -> colored::ColoredString {
    if p >= 0.9 {
        text.normal()
    } else {
        let (r, g, b) = heat(p);
        text.truecolor(r, g, b)
    }
}

/// Colored terminal report. Honors `NO_COLOR` through the `colored` crate.
pub fn terminal(tokens: &[LogprobToken], report: &Report, meta: &Meta) -> String {
    let mut out = String::new();
    let spans = span_index(tokens, report);

    let mut head = vec![meta.source.clone()];
    if let Some(m) = &meta.model {
        head.push(m.clone());
    }
    head.push(format!("{} tokens", report.token_count));
    head.push(format!("mean p {:.2}", report.mean_prob));
    head.push(format!("threshold {:.2}", report.threshold));
    let _ = writeln!(out, "\n  {}\n", head.join("   ").dimmed());

    // The answer as a heatmap, wrapped at about 72 columns.
    let mut line = String::from("  ");
    let mut col = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        let len = t.token.chars().count();
        let text = if col + len > 72 && t.token.starts_with(' ') {
            out.push_str(&line);
            out.push('\n');
            line = String::from("  ");
            col = 0;
            t.token.trim_start().to_string()
        } else {
            t.token.clone()
        };
        col += text.chars().count();
        // Keep a flagged word's leading space out of its underline.
        let body = text.trim_start();
        line.push_str(&text[..text.len() - body.len()]);
        let mut piece = piece_for(body, t.prob());
        if spans[i] > 0 {
            piece = piece.bold().underline();
        }
        line.push_str(&piece.to_string());
    }
    out.push_str(&line);
    out.push_str("\n\n");

    // One label width for every span so the bars line up.
    let width = report
        .spans
        .iter()
        .flat_map(candidates)
        .map(|c| show(&c.0).chars().count())
        .max()
        .unwrap_or(0);
    for (n, s) in report.spans.iter().enumerate() {
        let (r, g, b) = heat(s.min_prob);
        let _ = writeln!(
            out,
            "  {}  {}   {}",
            format!("{}", n + 1).truecolor(r, g, b).bold(),
            s.text.trim().bold(),
            format!(
                "{}, weakest {} at p {:.2}",
                token_range(s),
                show(&s.weakest_token),
                s.min_prob
            )
            .dimmed()
        );
        for (tok, p, chosen) in candidates(s) {
            let filled = (p * 24.0).round() as usize;
            let (r, g, b) = heat(p);
            let bar = format!(
                "{}{}",
                "\u{2588}".repeat(filled).truecolor(r, g, b),
                "\u{2591}".repeat(24 - filled.min(24)).dimmed()
            );
            let label = format!("{:<width$}", show(&tok));
            let note = if chosen { "  chosen" } else { "" };
            let _ = writeln!(
                out,
                "     {}  {}  {:.2}{}",
                if chosen { label.bold() } else { label.normal() },
                bar,
                p,
                note.dimmed()
            );
        }
        out.push('\n');
    }

    let msg = summary(report);
    let _ = writeln!(
        out,
        "  {}",
        if report.flagged() {
            msg.truecolor(240, 140, 52)
        } else {
            msg.normal()
        }
    );
    out
}

// ---------------------------------------------------------------- markdown

fn md_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        if "\\`*_[]<>|#~".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

/// Markdown report: flagged spans in bold with numbered markers, then a table.
pub fn markdown(tokens: &[LogprobToken], report: &Report, meta: &Meta) -> String {
    let mut md = String::new();
    let spans = span_index(tokens, report);
    let _ = writeln!(md, "### Token confidence report\n");
    let mut facts = vec![format!("`{}`", meta.source)];
    if let Some(m) = &meta.model {
        facts.push(format!("`{m}`"));
    }
    facts.push(format!("{} tokens", report.token_count));
    facts.push(format!("mean p {:.2}", report.mean_prob));
    facts.push(format!("threshold {:.2}", report.threshold));
    let _ = writeln!(md, "{}\n", facts.join(" · "));

    let mut body = String::new();
    let mut i = 0;
    while i < tokens.len() {
        if spans[i] == 0 {
            body.push_str(&md_escape(&tokens[i].token));
            i += 1;
            continue;
        }
        let n = spans[i];
        let s = &report.spans[n - 1];
        let text: String = tokens[s.start..s.end]
            .iter()
            .map(|t| t.token.as_str())
            .collect();
        let lead = &text[..text.len() - text.trim_start().len()];
        let _ = write!(
            body,
            "{lead}**{}**<sup>{n}</sup>",
            md_escape(text.trim_start())
        );
        i = s.end;
    }
    let _ = writeln!(md, "> {}\n", body.trim());

    if report.flagged() {
        md.push_str("| # | Span | Weakest token | p | Model also considered |\n");
        md.push_str("|---|---|---|---|---|\n");
        for (n, s) in report.spans.iter().enumerate() {
            let others: Vec<String> = candidates(s)
                .into_iter()
                .filter(|c| !c.2)
                .map(|c| format!("`{}` {:.2}", c.0.replace('`', "'"), c.1))
                .collect();
            let _ = writeln!(
                md,
                "| {} | {} | `{}` | {:.2} | {} |",
                n + 1,
                md_escape(s.text.trim()),
                s.weakest_token.replace('`', "'"),
                s.min_prob,
                if others.is_empty() {
                    "n/a".to_string()
                } else {
                    others.join(", ")
                }
            );
        }
        md.push('\n');
    }
    let _ = writeln!(md, "{}", summary(report));
    md
}

// -------------------------------------------------------------------- html

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            _ => o.push(c),
        }
    }
    o
}

const CSS: &str = r#"
:root{--bg:#f4efe4;--card:#fbf8f1;--ink:#1d1b17;--mute:#6f6a5f;--rule:#ddd5c4;--flag:#d9432b;--alpha:.62}
@media (prefers-color-scheme:dark){:root{--bg:#141311;--card:#1b1a17;--ink:#ebe5d8;--mute:#9a9385;--rule:#302d27;--flag:#f0654a;--alpha:.5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:48px 20px 64px}
.k{font:12px/1.4 ui-monospace,"SF Mono","Cascadia Mono",Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}
.meta{text-transform:none;letter-spacing:.02em;display:flex;flex-wrap:wrap;gap:6px 18px;margin:10px 0 28px}
.meta b{color:var(--ink);font-weight:600}
.answer{font:clamp(22px,3.4vw,30px)/1.75 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;margin:0 0 20px;padding:28px 30px;background:var(--card);border:1px solid var(--rule);border-radius:14px}
.t{position:relative;border-radius:4px;background:rgba(var(--h),calc(var(--a)*var(--alpha)));padding:.06em 0}
.t:hover,.t:focus{outline:1.5px solid rgb(var(--h));outline-offset:1px}
.t:hover::after,.t:focus::after{content:attr(data-tip);position:absolute;left:0;top:calc(100% + 6px);z-index:2;white-space:pre;background:var(--ink);color:var(--bg);font:12px/1.5 ui-monospace,Consolas,monospace;padding:6px 9px;border-radius:6px;pointer-events:none}
.f{text-decoration:underline wavy var(--flag);text-decoration-thickness:1.5px;text-underline-offset:.28em;text-decoration-skip-ink:none}
sup{font:600 11px ui-monospace,Consolas,monospace;color:var(--flag);margin-left:1px}
.trace{justify-content:center;display:flex;align-items:flex-end;gap:2px;height:44px;margin:0 0 36px;padding:0 4px}
.trace i{flex:1;max-width:26px;min-width:2px;border-radius:2px 2px 0 0;background:rgb(var(--h))}
.spans{display:grid;gap:14px}
.span{background:var(--card);border:1px solid var(--rule);border-radius:12px;padding:16px 18px}
.span h3{margin:0 0 10px;font-size:17px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.span h3 .k{text-transform:none;letter-spacing:0}
.span h3 .n{font:600 13px ui-monospace,Consolas,monospace;color:var(--flag)}
.row{display:grid;grid-template-columns:minmax(90px,max-content) 1fr 44px;gap:12px;align-items:center;font:14px ui-monospace,Consolas,monospace;margin:4px 0}
.bar{height:10px;border-radius:5px;background:var(--rule);overflow:hidden}
.bar i{display:block;height:100%;background:rgb(var(--h))}
.row.c{font-weight:700}
.p{text-align:right;color:var(--mute)}
.verdict{margin:30px 0 0;padding-top:18px;border-top:1px solid var(--rule)}
.scale{display:flex;align-items:center;gap:10px;margin-top:14px}
.scale span.g{height:8px;width:180px;border-radius:4px;background:linear-gradient(90deg,rgb(232,72,48),rgb(240,140,52) 40%,rgb(236,192,92) 76%,rgb(214,208,196))}
footer{margin-top:40px}
footer a{color:inherit}
"#;

/// Self-contained HTML report. Hover or focus any token for its probability
/// and the alternatives the model weighed.
pub fn html(tokens: &[LogprobToken], report: &Report, meta: &Meta) -> String {
    let spans = span_index(tokens, report);
    let mut h = String::new();
    let _ = write!(
        h,
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <meta name=\"color-scheme\" content=\"light dark\">\
         <title>Token confidence: {}</title><style>{}</style></head><body><main>",
        esc(&meta.source),
        CSS
    );
    let _ = write!(h, "<div class=\"k\">Token confidence report</div>");
    let _ = write!(
        h,
        "<div class=\"meta k\"><span><b>{}</b></span>",
        esc(&meta.source)
    );
    if let Some(m) = &meta.model {
        let _ = write!(h, "<span>{}</span>", esc(m));
    }
    let _ = write!(
        h,
        "<span>{} tokens</span><span>mean p {:.2}</span><span>threshold {:.2}</span>\
         <span>{} flagged</span></div>",
        report.token_count,
        report.mean_prob,
        report.threshold,
        report.spans.len()
    );

    // The answer, one element per token.
    h.push_str("<p class=\"answer\">");
    for (i, t) in tokens.iter().enumerate() {
        let n = spans[i];
        let starts = n > 0 && (i == 0 || spans[i - 1] != n);
        let ends = n > 0 && (i + 1 == tokens.len() || spans[i + 1] != n);
        let body = t.token.trim_start();
        let lead = &t.token[..t.token.len() - body.len()];
        h.push_str(&esc(lead));
        if starts {
            let _ = write!(h, "<span class=\"f\" id=\"s{n}\">");
        }
        let p = t.prob();
        let (r, g, b) = heat(p);
        let mut tip = format!("p {p:.2}  {}", show(&t.token));
        for a in t.top_logprobs.iter().filter(|a| a.token != t.token) {
            let _ = write!(tip, "\n     {:.2}  {}", a.logprob.exp(), show(&a.token));
        }
        let _ = write!(
            h,
            "<span class=\"t\" tabindex=\"0\" style=\"--h:{r},{g},{b};--a:{:.3}\" data-tip=\"{}\">{}</span>",
            ((0.95 - p) / 0.95).max(0.0).powf(0.7),
            esc(&tip),
            esc(body)
        );
        if ends {
            let _ = write!(h, "</span><sup>{n}</sup>");
        }
    }
    h.push_str("</p>");

    // A strip with one bar per token, height = probability.
    h.push_str("<div class=\"trace\" aria-hidden=\"true\">");
    for t in tokens {
        let p = t.prob();
        let (r, g, b) = heat(p);
        let _ = write!(
            h,
            "<i style=\"--h:{r},{g},{b};height:{:.0}%\" title=\"{} p {:.2}\"></i>",
            (p * 100.0).max(4.0),
            esc(&show(&t.token)),
            p
        );
    }
    h.push_str("</div>");

    if report.flagged() {
        h.push_str("<div class=\"k\" style=\"margin-bottom:12px\">Flagged spans</div><div class=\"spans\">");
        for (n, s) in report.spans.iter().enumerate() {
            let _ = write!(
                h,
                "<section class=\"span\"><h3><span class=\"n\">{}</span>{}<span class=\"k\">weakest {} at p {:.2}</span></h3>",
                n + 1,
                esc(s.text.trim()),
                esc(&show(&s.weakest_token)),
                s.min_prob
            );
            for (tok, p, chosen) in candidates(s) {
                let (r, g, b) = heat(p);
                let _ = write!(
                    h,
                    "<div class=\"row{}\"><span>{}</span><span class=\"bar\"><i style=\"--h:{r},{g},{b};width:{:.1}%\"></i></span><span class=\"p\">{:.2}</span></div>",
                    if chosen { " c" } else { "" },
                    esc(&show(&tok)),
                    p * 100.0,
                    p
                );
            }
            h.push_str("</section>");
        }
        h.push_str("</div>");
    }

    let _ = writeln!(
        h,
        "<div class=\"verdict\">{}<div class=\"scale k\"><span>p 0</span><span class=\"g\"></span><span>p 1</span></div></div>\
         <footer class=\"k\">Generated by <a href=\"https://github.com/Mattbusel/LLM-Hallucination-Detection-Script\">llm-token-visualizer</a> {}</footer>\
         </main></body></html>\n",
        esc(&summary(report)),
        env!("CARGO_PKG_VERSION")
    );
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::{detect, parse_logprobs};

    fn sample() -> (Vec<LogprobToken>, Report, Meta) {
        let json = include_str!("../examples/logprobs/cuyp.json");
        let tokens = parse_logprobs(json).unwrap();
        let report = detect(&tokens, 0.6);
        (tokens, report, Meta::from_response("cuyp.json", json))
    }

    #[test]
    fn meta_reads_model_name() {
        let (_, _, m) = sample();
        assert_eq!(m.model.as_deref(), Some("meta-llama/llama-3.1-8b-instruct"));
    }

    #[test]
    fn heat_is_monotone_from_red_to_ink() {
        assert_eq!(heat(0.0), (232, 72, 48));
        assert_eq!(heat(1.0), (214, 208, 196));
        // green channel rises as confidence rises
        let gs: Vec<u8> = [0.2, 0.4, 0.6, 0.8, 0.95]
            .iter()
            .map(|&p| heat(p).1)
            .collect();
        assert!(gs.windows(2).all(|w| w[0] <= w[1]), "{gs:?}");
    }

    #[test]
    fn html_marks_spans_and_alternatives() {
        let (t, r, m) = sample();
        let h = html(&t, &r, &m);
        assert!(h.starts_with("<!DOCTYPE html>"));
        assert!(h.contains("id=\"s2\""));
        assert!(h.contains("&quot;üsseldorf&quot;"));
        assert!(h.contains("2 low-confidence spans"));
        assert_eq!(h.matches("class=\"t\"").count(), t.len());
        assert!(!h.contains("<script"));
    }

    #[test]
    fn html_escapes_token_text() {
        let tokens = parse_logprobs(r#"[{"token":"<b>&","logprob":-2.0}]"#).unwrap();
        let r = detect(&tokens, 0.5);
        let h = html(&tokens, &r, &Meta::default());
        assert!(h.contains("&lt;b&gt;&amp;"));
        assert!(!h.contains("<b>&"));
    }

    #[test]
    fn markdown_numbers_spans_and_lists_alternatives() {
        let (t, r, m) = sample();
        let md = markdown(&t, &r, &m);
        assert!(md.contains("**Dordrecht**<sup>2</sup>"), "{md}");
        assert!(md.contains("`üsseldorf` 0.39"), "{md}");
        assert!(md.contains("| 2 | Dordrecht | `ord` | 0.57 |"), "{md}");
    }

    #[test]
    fn terminal_without_color_is_plain_and_complete() {
        colored::control::set_override(false);
        let (t, r, m) = sample();
        let out = terminal(&t, &r, &m);
        assert!(out.contains("Aelbert Cuyp died in 1691 in Dordrecht, Netherlands."));
        assert!(out.contains("\"üsseldorf\""));
        assert!(out.contains("chosen"));
        assert!(out.contains("2 low-confidence spans"));
        colored::control::unset_override();
    }

    #[test]
    fn confident_answer_still_warns() {
        let tokens = parse_logprobs(r#"[{"token":"Yes","logprob":-0.01}]"#).unwrap();
        let r = detect(&tokens, 0.5);
        assert!(markdown(&tokens, &r, &Meta::default()).contains("confidently wrong"));
    }
}
