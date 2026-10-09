# LLM 幻觉检测器

[English](README.md) | 简体中文 | [日本語](README.ja.md) | [한국어](README.ko.md)

**把 AI 聊天机器人没把握的词高亮出来，让你知道它的回答里哪些地方需要再核实一遍。**

适合所有上线或审核大模型（LLM）回答的人：开发者、评测人员，以及 CI 流水线。支持 OpenAI、OpenRouter、Together、vLLM、Ollama，以及任何返回 token logprobs 的 OpenAI 兼容 API。提供 Rust 命令行工具和库（`llm-token-visualizer`）。

支持任何语言的回答：词语按 Unicode 分词规则切分，所以中文或日文回答里哪怕只有一个字没把握，也会被单独标出来。可以读取 OpenAI 风格、completions 风格以及 Google Gemini（`logprobsResult`）的响应。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.png">
  <img alt="对 Llama 3.1 一个真实回答做 LLM 幻觉检测：'Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.' 以 token 置信度热力图显示，Aelbert 和 Dordrecht 被标出，'ord' 处的候选为：ord 0.57、üsseldorf 0.39、elf 0.03。" src="docs/assets/hero-light.png" width="100%">
</picture>

<p align="center">
  <a href="https://crates.io/crates/llm-token-visualizer"><img alt="crates.io" src="https://img.shields.io/crates/v/llm-token-visualizer.svg"></a>
</p>

## 安装

**Linux**（x86_64，Ubuntu 20.04+ / Debian 11+）。一行命令，无需任何依赖，安装到 `~/.local/bin`：

```sh
mkdir -p ~/.local/bin && curl -fsSL https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-linux-x86_64.tar.gz | tar xz --strip-components=1 -C ~/.local/bin --wildcards '*/llm-token-visualizer'
```

| 其他系统 | |
|---|---|
| **Windows** | [下载 llm-token-visualizer-windows-x86_64.exe](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-windows-x86_64.exe) 后直接运行。（程序未签名，SmartScreen 可能会弹出提示：先点 *更多信息*，再点 *仍要运行*。） |
| **任何装了 Rust 的系统** | `cargo binstall llm-token-visualizer`（提供 Linux 和 Windows 预编译二进制）或 `cargo install --locked llm-token-visualizer` |
| **作为库使用** | `cargo add llm-token-visualizer --no-default-features`（不含 HTTP 客户端） |

发布包里还附带了 `samples/` 下的示例回答。所有版本及其 SHA-256 校验和：[Releases](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases)。

**在 GitLab CI 中**：用 [`hallucination-gate`](https://gitlab.com/explore/catalog/mattbusel/llm-ci) CI/CD 组件，当保存下来的 LLM 回答里有低置信度的词时，让流水线失败。

## 工作原理

<img alt="动画示意图，演示检测器如何处理一个真实回答。第 1 步：每个 token 都带有一个 logprob，p 等于 e 的 logprob 次方，每个 token 按颜色从红色（没把握）到灰色（有把握）着色；'A' 的 p 为 0.49，'ord' 的 p 为 0.57。第 2 步：token 拼成词，只要词中任一 token 低于 0.6 的阈值，这个词就会被标出，因此 Aelbert 和 Dordrecht 被标出。第 3 步：每个被标出的片段都会显示模型考虑过的候选：The 0.43 而不是 A，üsseldorf 0.39 而不是 ord。" src="docs/img/how-it-works.svg" width="100%">

模型写回答时，每个 token（一个词或词的一部分）都是从一组候选里挑出来的，每个候选都有一个概率。API 以 `logprobs` 的形式提供这些概率。幻觉往往就出现在概率下降的地方：一个人名、一个日期、一座模型记得不太清的城市。这个工具把这些数字变成热力图，标出不稳的词，并列出模型差点说出口的候选。它自己不带模型，除非你用 `--live`，否则不会调用任何 API。

## 示例

下面是 Llama 3.1 8B Instruct 的真实回答，随仓库附带在 `examples/logprobs/` 中。这是 `cargo run --example detect`（阈值 0.6）输出的四个回答中的前两个：

```text
examples/logprobs/cuyp.json
  answer: Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.
  flagged "Aelbert": p=0.49 at "A"; model also considered "The" (0.43), "D" (0.08)
  flagged "Dordrecht": p=0.57 at "ord"; model also considered "üsseldorf" (0.39), "elf" (0.03)

examples/logprobs/tour-de-france.json
  answer: Stephen Roche won the 1987 Tour de France, riding for the Carrera Jeans-Vagabond team.
  flagged "Stephen": p=0.49 at "Stephen"; model also considered "The" (0.43), "Steven" (0.03)
  flagged "-Vagabond": p=0.57 at "-V"; model also considered "–" (0.21), " -" (0.14)
```

Dordrecht 这个回答是对的，但模型给了 Düsseldorf 39% 的概率：这正是需要再核实一遍的那种说法。"Aelbert" 被标出，是因为模型在"以人名开头"和"以 The 开头"之间犹豫：低概率可能只是措辞问题，而不是事实问题。

第一个回答在终端里的报告（`--logprobs-file examples/logprobs/cuyp.json --threshold 0.6`）：

<img alt="llm-token-visualizer 处理 examples/logprobs/cuyp.json 的终端输出：回答中 Aelbert 和 Dordrecht 带下划线，下面是每个弱 token 处候选的条形图：A 0.49、The 0.43、D 0.08，以及 ord 0.57、üsseldorf 0.39、elf 0.03。" src="docs/assets/terminal-cuyp.png" width="720">

### 它无法告诉你的事

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/confidently-wrong-dark.png">
  <img alt="moonwalk 示例：'Pete Conrad was the second person to walk on the Moon'。Pete Conrad 被圈出表示错误，但没有被标出：P 0.72、ete 0.76、Conrad 1.00。被标出的反而是：was 0.60、which 0.14、during 0.55。" src="docs/assets/confidently-wrong-light.png" width="100%">
</picture>

模型也可能自信地答错。在 `moonwalk.json` 里，模型说"Pete Conrad was the second person to walk on the Moon"（实际上是 Buzz Aldrin），而这个名字的每个 token 概率都至少有 72%，所以错误的名字没有被标出。用它来决定先看哪里，而不是用它来证明一个回答是对的。

## 为什么用这个工具

目前没有其他基于 logprob 做置信度检查的 Rust crate。Python 里的 [LM-Polygraph](https://github.com/IINemo/lm-polygraph) 和 [UQLM](https://github.com/cvs-health/uqlm) 提供了多得多的不确定性方法（采样一致性、声明级评分、训练好的估计器），但需要加载模型，或者每个回答生成多次。这个工具只用一份保存下来的响应做一件便宜的事：显示哪些词概率低、模型还考虑过什么，输出形式可以是终端热力图、HTML 页面、用于合并请求的 Markdown、JSON，或者 CI 退出码。每份报告还会给出回答的困惑度（perplexity），每个被标出的片段还会给出其最弱 token 处候选的熵。

## 3 步上手

1. **获取带 logprobs 的响应**：让你的 API 返回 `"logprobs": true, "top_logprobs": 3` 并保存 JSON，或者让工具代劳：设置 `OPENAI_API_KEY` 后运行 `llm-token-visualizer --live "your question" --save answer.json`。
2. **运行**：`llm-token-visualizer --logprobs-file answer.json --threshold 0.6`
3. **分享或设为门禁**：用 `--format html -o report.html` 生成一个可以发给别人的页面，用 `--format markdown` 生成 PR 评论，用 `--fail-on-flag` 在有任何内容被标出时让 CI 任务失败。

## 服务商

`--live` 会向模型提问并分析它的回答。用 `--provider` 选择服务商（默认 `openai`），用 `--base-url` 覆盖地址：

| `--provider` | 默认 base URL | 密钥来源 | 默认 `--model` | 验证方式 |
|---|---|---|---|---|
| `openai` | `https://api.openai.com/v1`（或 `OPENAI_BASE_URL`） | `OPENAI_API_KEY` | `gpt-4o-mini` | 文档（`logprobs`，`top_logprobs` 最多 5）。请求构建有单元测试。 |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | 文档（`logprobs`、`top_logprobs`；是否支持取决于上游服务商，所以请求会设置 `provider.require_parameters`）。请求构建有单元测试。 |
| `together` | `https://api.together.ai/v1` | `TOGETHER_API_KEY` | 无，需传 `--model` | 文档（Together 接受的是 `"logprobs": <int>` 而不是 `true`，CLI 会按此发送）。请求构建有单元测试。 |
| `vllm` | `http://localhost:8000/v1` | `VLLM_API_KEY`，仅当服务端使用了 `--api-key` 时需要 | 无，需传 `--model` | 文档（Chat Completions 中的 `logprobs`、`top_logprobs`）。请求构建有单元测试。 |
| `ollama` | `http://localhost:11434/v1` | 无需密钥 | 无，需传 `--model` | 用 `qwen2.5-coder:14b` 对 Ollama 0.34.4 做了**真实调用**：返回了 logprobs 和前 3 个候选。（Ollama 自己的 OpenAI 兼容性页面仍把 logprobs 列为不支持；旧版本可能不会返回。） |

```sh
llm-token-visualizer --live "Who painted The Night Watch?" --provider ollama --model qwen2.5-coder:14b
llm-token-visualizer --live "Who painted The Night Watch?" --provider openrouter --save answer.json
llm-token-visualizer --live "..." --provider vllm --model my-model --base-url http://gpu-box:8000/v1
```

本次发布没有用真实密钥调用过任何托管服务商。如果某个服务商或模型的回答不带 logprobs，CLI 会报错说明这一点并显示回答，而不是打印一份空报告。Anthropic 的 API 不返回 logprobs，所以没有为它提供预设。

Google Gemini 的原生 API 也会返回 logprobs（`generationConfig: {responseLogprobs: true, logprobs: 3}`）；把响应保存下来，用 `--logprobs-file` 传入即可。

手头没有 API 密钥？先下载一个示例：`curl -LO https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/raw/main/examples/logprobs/cuyp.json`

## 库

```rust
use llm_token_visualizer::detect::{detect, parse_logprobs};

let json = std::fs::read_to_string("examples/logprobs/cuyp.json")?;
let report = detect(&parse_logprobs(&json)?, 0.6);
for span in &report.spans {
    println!("{:?}: {}", span.text.trim(), span.describe());
}
println!("perplexity {:.2}", report.perplexity);
# Ok::<(), anyhow::Error>(())
```

| Feature | 默认 | 新增内容 |
|---|---|---|
| `live` | 开启 | `--live` 和 `live` 模块：向 OpenAI 兼容 API 提问（ureq、rustls；超时 120 秒） |
| `async-openai` | 关闭 | `interop::async_openai::tokens_from_response`，用于 `async_openai` 的 chat completion 响应（仅类型，不含 HTTP 客户端） |

示例（`cargo run --example <name>`）：`detect`（所有附带的示例）、`ci_gate`（一个装满回答的目录，只要有任何内容被标出就以 2 退出）、`html_report`（为一个回答生成 HTML 页面）。

已经在用 **async-openai** 了？启用这个 feature，把你的 `CreateChatCompletionResponse` 传给 `interop::async_openai::tokens_from_response`，再调用 `detect`。用的是别的客户端？把它的响应序列化成 JSON，然后调用 `parse_logprobs`。

## 性能

`cargo bench --bench vs_0_4` 解析一个 4,000 token 的响应并标出片段：0.5.0 用时 9.0 ms，0.4.0 用时 14.6 ms（criterion 中位数，i7-13700KF，Windows 11，Rust 1.91）。0.5.0 反序列化 logprobs 时不再先复制一份 JSON，省下的时间足以抵消 Unicode 分词的开销还有余。

## 文档

| | |
|---|---|
| [参考手册](docs/REFERENCE.md) | 所有参数、输出格式（终端、HTML、Markdown、JSON）、CI 用法、输入格式、live 模式、自定义置信度分数、库 API |
| [工作原理与仓库结构](docs/ARCHITECTURE.md) | 检测规则和配色方案的细节、源码结构、哪些只是草稿、哪些已正式发布 |
| [docs.rs 上的 API 文档](https://docs.rs/llm-token-visualizer) | Rust 库 |
| [项目网站](https://hallucination-highlighter.vercel.app/) | 概览 |
| [浏览器版（源码）](docs/try/index.html) | 带阈值滑块的示例，或用你自己的密钥向服务商提问；从克隆下来的仓库里打开该文件 |
| [更新日志](CHANGELOG.md) | 每个版本的改动 |

## 许可证

MIT，见 [LICENSE](LICENSE)。
