# LLM 환각 탐지기

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | 한국어

**AI 챗봇이 확신하지 못한 단어를 하이라이트해서, 답변의 어느 부분을 다시 확인해야 하는지 알려 줍니다.**

LLM 답변을 서비스에 내보내거나 검수하는 모든 분을 위한 도구입니다: 개발자, 평가자, 그리고 CI 파이프라인. OpenAI, OpenRouter, Together, vLLM, Ollama, 그리고 토큰 logprobs를 반환하는 모든 OpenAI 호환 API에서 동작합니다. Rust CLI이자 라이브러리(`llm-token-visualizer`)입니다.

어떤 언어로 된 답변이든 쓸 수 있습니다. 단어를 Unicode 단어 분할로 찾기 때문에, 중국어나 일본어 답변에서 확신 없는 글자 하나만 따로 표시됩니다. OpenAI 형식, completions 형식, Google Gemini(`logprobsResult`) 응답을 읽습니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.png">
  <img alt="Llama 3.1의 실제 답변 'Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.'에 대한 LLM 환각 탐지. 토큰 신뢰도 히트맵으로 표시되며 Aelbert와 Dordrecht가 표시되었고, 'ord' 위치의 후보는 ord 0.57, üsseldorf 0.39, elf 0.03." src="docs/assets/hero-light.png" width="100%">
</picture>

<p align="center">
  <a href="https://crates.io/crates/llm-token-visualizer"><img alt="crates.io" src="https://img.shields.io/crates/v/llm-token-visualizer.svg"></a>
</p>

## 설치

**Linux**(x86_64, Ubuntu 20.04+ / Debian 11+). 의존성 없이 한 줄로 `~/.local/bin`에 설치합니다:

```sh
mkdir -p ~/.local/bin && curl -fsSL https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-linux-x86_64.tar.gz | tar xz --strip-components=1 -C ~/.local/bin --wildcards '*/llm-token-visualizer'
```

| 다른 시스템 | |
|---|---|
| **Windows** | [llm-token-visualizer-windows-x86_64.exe 다운로드](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-windows-x86_64.exe) 후 실행하세요. (서명되지 않은 파일이라 SmartScreen이 물어볼 수 있습니다: *추가 정보*를 누른 다음 *실행*을 누르세요.) |
| **Rust가 설치된 모든 시스템** | `cargo binstall llm-token-visualizer`(Linux와 Windows용 사전 빌드 바이너리) 또는 `cargo install --locked llm-token-visualizer` |
| **라이브러리로 사용** | `cargo add llm-token-visualizer --no-default-features`(HTTP 클라이언트 없음) |

릴리스 아카이브에는 `samples/` 아래의 샘플 답변도 들어 있습니다. SHA-256 체크섬이 포함된 모든 릴리스: [Releases](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases).

**GitLab CI에서**: [`hallucination-gate`](https://gitlab.com/explore/catalog/mattbusel/llm-ci) CI/CD 컴포넌트로, 저장된 LLM 답변에 신뢰도가 낮은 단어가 있으면 파이프라인을 실패시킬 수 있습니다.

## 동작 원리

<img alt="탐지기가 실제 답변을 처리하는 과정을 보여 주는 애니메이션 다이어그램. 1단계: 모든 토큰에는 logprob이 있고 p는 e의 logprob 제곱이며, 각 토큰은 빨간색(확신 없음)부터 회색(확신 있음)까지 색으로 표시된다. 'A'의 p는 0.49, 'ord'의 p는 0.57. 2단계: 토큰이 단어로 합쳐지고, 단어 안의 토큰 중 하나라도 임계값 0.6보다 낮으면 그 단어가 표시되므로 Aelbert와 Dordrecht가 표시된다. 3단계: 표시된 각 구간에는 모델이 고려했던 후보가 나온다: A 대신 The 0.43, ord 대신 üsseldorf 0.39." src="docs/img/how-it-works.svg" width="100%">

모델은 답변을 쓸 때 각 토큰(단어 또는 단어의 일부)을 후보 목록에서 고르며, 후보마다 확률이 있습니다. API는 이것을 `logprobs`로 제공합니다. 환각은 이 확률이 떨어지는 곳에 자주 숨어 있습니다: 이름, 날짜, 모델이 어렴풋이 기억하는 도시 같은 것들이죠. 이 도구는 그 숫자를 히트맵으로 바꾸고 불안한 단어를 표시하며, 모델이 말할 뻔했던 대안도 함께 보여 줍니다. 자체 모델은 없고, `--live`를 요청하지 않는 한 API도 호출하지 않습니다.

## 예시

Llama 3.1 8B Instruct의 실제 답변으로, `examples/logprobs/`에 포함되어 있습니다. `cargo run --example detect`(임계값 0.6) 출력에 나오는 네 개의 답변 중 처음 두 개입니다:

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

Dordrecht 답변은 맞지만, 모델은 Düsseldorf에 39%의 확률을 주었습니다: 바로 다시 확인해야 할 종류의 주장입니다. "Aelbert"가 표시된 것은 모델이 이름으로 시작할지 "The"로 시작할지 고민했기 때문입니다: 낮은 확률은 사실이 아니라 표현 방식의 문제일 수도 있습니다.

첫 번째 답변의 터미널 리포트(`--logprobs-file examples/logprobs/cuyp.json --threshold 0.6`):

<img alt="examples/logprobs/cuyp.json에 대한 llm-token-visualizer의 터미널 출력: Aelbert와 Dordrecht에 밑줄이 그어진 답변, 그 아래로 약한 토큰마다 후보의 막대그래프: A 0.49, The 0.43, D 0.08, 그리고 ord 0.57, üsseldorf 0.39, elf 0.03." src="docs/assets/terminal-cuyp.png" width="720">

### 이 도구가 알려 줄 수 없는 것

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/confidently-wrong-dark.png">
  <img alt="moonwalk 샘플: 'Pete Conrad was the second person to walk on the Moon'. Pete Conrad는 틀린 것으로 테두리가 쳐져 있지만 표시되지 않았다: P 0.72, ete 0.76, Conrad 1.00. 대신 표시된 것: was 0.60, which 0.14, during 0.55." src="docs/assets/confidently-wrong-light.png" width="100%">
</picture>

모델은 자신만만하게 틀릴 수 있습니다. `moonwalk.json`에서 모델은 "Pete Conrad was the second person to walk on the Moon"이라고 답하는데(정답은 Buzz Aldrin), 이름의 모든 토큰에 최소 72%의 확률을 주었기 때문에 틀린 이름이 표시되지 않습니다. 답변을 보증하는 용도가 아니라, 어디부터 확인할지 정하는 용도로 쓰세요.

## 왜 이 도구인가

logprob 기반으로 신뢰도를 검사하는 Rust 크레이트는 이것 말고는 없습니다. Python에서는 [LM-Polygraph](https://github.com/IINemo/lm-polygraph)와 [UQLM](https://github.com/cvs-health/uqlm)이 훨씬 많은 불확실성 측정 방법(샘플링 일관성, 주장 단위 점수, 학습된 추정기)을 제공하지만, 모델을 직접 로드하거나 답변 하나당 여러 번 생성해야 합니다. 이 도구는 저장된 응답 하나로 저렴한 일 하나만 합니다: 어떤 단어의 확률이 낮았는지, 모델이 그 밖에 무엇을 고려했는지를 터미널 히트맵, HTML 페이지, 머지 리퀘스트용 Markdown, JSON, 또는 CI 종료 코드로 보여 줍니다. 각 리포트에는 답변의 퍼플렉시티(perplexity)도, 표시된 각 구간에는 가장 약한 토큰에서의 후보 엔트로피도 나옵니다.

## 3단계로 사용하기

1. **logprobs가 포함된 응답 받기**: API에 `"logprobs": true, "top_logprobs": 3`을 요청하고 JSON을 저장하세요. 또는 도구에 맡길 수도 있습니다: `OPENAI_API_KEY`를 설정하고 `llm-token-visualizer --live "your question" --save answer.json`을 실행하세요.
2. **실행하기**: `llm-token-visualizer --logprobs-file answer.json --threshold 0.6`
3. **공유하거나 게이트로 쓰기**: 보낼 수 있는 페이지는 `--format html -o report.html`, PR 댓글은 `--format markdown`, 무엇이든 표시되면 CI 작업을 실패시키려면 `--fail-on-flag`.

## 프로바이더

`--live`는 모델에 질문하고 그 답변을 분석합니다. 어디에 물을지는 `--provider`로 고르고(기본값 `openai`), 주소는 `--base-url`로 덮어쓸 수 있습니다:

| `--provider` | 기본 base URL | 키 출처 | 기본 `--model` | 확인 방법 |
|---|---|---|---|---|
| `openai` | `https://api.openai.com/v1`(또는 `OPENAI_BASE_URL`) | `OPENAI_API_KEY` | `gpt-4o-mini` | 문서(`logprobs`, `top_logprobs` 최대 5). 요청 생성은 단위 테스트됨. |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | 문서(`logprobs`, `top_logprobs`; 지원 여부는 업스트림 프로바이더에 따라 다르므로 요청에 `provider.require_parameters`를 설정). 요청 생성은 단위 테스트됨. |
| `together` | `https://api.together.ai/v1` | `TOGETHER_API_KEY` | 없음, `--model` 지정 | 문서(Together는 `true` 대신 `"logprobs": <int>`를 받으며, CLI가 그렇게 보냄). 요청 생성은 단위 테스트됨. |
| `vllm` | `http://localhost:8000/v1` | `VLLM_API_KEY`, 서버가 `--api-key`를 쓸 때만 | 없음, `--model` 지정 | 문서(Chat Completions의 `logprobs`, `top_logprobs`). 요청 생성은 단위 테스트됨. |
| `ollama` | `http://localhost:11434/v1` | 키 없음 | 없음, `--model` 지정 | `qwen2.5-coder:14b`로 Ollama 0.34.4에 **실제 호출**: logprobs와 상위 3개 후보가 돌아옴. (Ollama의 OpenAI 호환성 페이지에는 아직 logprobs가 미지원으로 나와 있으며, 이전 버전은 반환하지 않을 수 있음.) |

```sh
llm-token-visualizer --live "Who painted The Night Watch?" --provider ollama --model qwen2.5-coder:14b
llm-token-visualizer --live "Who painted The Night Watch?" --provider openrouter --save answer.json
llm-token-visualizer --live "..." --provider vllm --model my-model --base-url http://gpu-box:8000/v1
```

이번 릴리스에서는 호스팅 프로바이더를 실제 키로 호출해 보지 않았습니다. 프로바이더나 모델이 logprobs 없이 답하면, CLI는 빈 리포트를 출력하는 대신 그렇다고 알려 주는 오류와 함께 멈춥니다(답변은 보여 줍니다). Anthropic API는 logprobs를 반환하지 않으므로 프리셋이 없습니다.

Google Gemini 네이티브 API도 logprobs를 반환합니다(`generationConfig: {responseLogprobs: true, logprobs: 3}`). 응답을 저장해서 `--logprobs-file`로 넘기세요.

당장 API 키가 없다면? 먼저 샘플을 받아 보세요: `curl -LO https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/raw/main/examples/logprobs/cuyp.json`

## 라이브러리

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

| 피처 | 기본값 | 추가되는 것 |
|---|---|---|
| `live` | 켜짐 | `--live`와 `live` 모듈: OpenAI 호환 API에 질문(ureq, rustls; 타임아웃 120초) |
| `async-openai` | 꺼짐 | `async_openai` chat completion 응답용 `interop::async_openai::tokens_from_response`(타입만, HTTP 클라이언트 없음) |

예제(`cargo run --example <name>`): `detect`(포함된 모든 샘플), `ci_gate`(답변 디렉터리를 처리하고, 하나라도 표시되면 종료 코드 2), `html_report`(답변 하나의 HTML 페이지 작성).

이미 **async-openai**를 쓰고 있나요? 이 피처를 켜고 `CreateChatCompletionResponse`를 `interop::async_openai::tokens_from_response`에 넘긴 다음 `detect`를 호출하세요. 다른 클라이언트를 쓴다면 응답을 JSON으로 직렬화해서 `parse_logprobs`를 호출하면 됩니다.

## 성능

`cargo bench --bench vs_0_4`는 4,000토큰짜리 응답을 파싱하고 구간을 표시합니다: 0.5.0은 9.0 ms, 0.4.0은 14.6 ms입니다(criterion 중앙값, i7-13700KF, Windows 11, Rust 1.91). 0.5.0은 JSON을 먼저 복사하지 않고 logprobs를 역직렬화하는데, 그 덕분에 Unicode 단어 분할 비용을 내고도 남습니다.

## 문서

| | |
|---|---|
| [레퍼런스](docs/REFERENCE.md) | 모든 플래그, 출력 형식(터미널, HTML, Markdown, JSON), CI 사용법, 입력 형식, 라이브 모드, 직접 만든 신뢰도 점수, 라이브러리 API |
| [동작 원리와 저장소 구조](docs/ARCHITECTURE.md) | 탐지 규칙과 색상 스케일 상세, 소스 구조, 무엇이 스케치이고 무엇이 배포되는지 |
| [docs.rs의 API 문서](https://docs.rs/llm-token-visualizer) | Rust 라이브러리 |
| [프로젝트 사이트](https://hallucination-highlighter.vercel.app/) | 개요 |
| [브라우저 버전(소스)](docs/try/index.html) | 임계값 슬라이더가 있는 샘플, 또는 본인 키로 프로바이더에 질문; 클론한 저장소에서 파일을 여세요 |
| [변경 이력](CHANGELOG.md) | 릴리스별 변경 사항 |

## 라이선스

MIT, [LICENSE](LICENSE)를 참고하세요.
