# LLM ハルシネーション検出ツール

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語 | [한국어](README.ko.md)

**AI チャットボットが自信を持てなかった単語をハイライトし、回答のどこを確認し直すべきかを示します。**

LLM の回答を本番に出す人、チェックする人すべてに：開発者、評価担当者、そして CI パイプライン。OpenAI、OpenRouter、Together、vLLM、Ollama、そしてトークンの logprobs を返すあらゆる OpenAI 互換 API で動きます。Rust 製の CLI とライブラリ（`llm-token-visualizer`）です。

どの言語の回答にも対応します。単語は Unicode の単語分割で切り出すので、中国語や日本語の回答で 1 文字だけ自信がなければ、その文字だけがフラグ付けされます。OpenAI 形式、completions 形式、Google Gemini（`logprobsResult`）のレスポンスを読み込めます。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.png">
  <img alt="Llama 3.1 の実際の回答 'Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.' に対する LLM ハルシネーション検出。トークン信頼度のヒートマップとして表示され、Aelbert と Dordrecht にフラグが付き、'ord' の位置の候補は ord 0.57、üsseldorf 0.39、elf 0.03。" src="docs/assets/hero-light.png" width="100%">
</picture>

<p align="center">
  <a href="https://crates.io/crates/llm-token-visualizer"><img alt="crates.io" src="https://img.shields.io/crates/v/llm-token-visualizer.svg"></a>
</p>

## インストール

**Linux**（x86_64、Ubuntu 20.04+ / Debian 11+）。依存関係なしの 1 行で、`~/.local/bin` にインストールします。

```sh
mkdir -p ~/.local/bin && curl -fsSL https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-linux-x86_64.tar.gz | tar xz --strip-components=1 -C ~/.local/bin --wildcards '*/llm-token-visualizer'
```

| その他の環境 | |
|---|---|
| **Windows** | [llm-token-visualizer-windows-x86_64.exe をダウンロード](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-windows-x86_64.exe)して実行するだけ。（署名なしのため SmartScreen が確認してくることがあります。*詳細情報* を押してから *実行* を押してください。） |
| **Rust が入っている環境** | `cargo binstall llm-token-visualizer`（Linux と Windows のビルド済みバイナリ）または `cargo install --locked llm-token-visualizer` |
| **ライブラリとして** | `cargo add llm-token-visualizer --no-default-features`（HTTP クライアントなし） |

リリースのアーカイブには `samples/` 以下のサンプル回答も入っています。全リリースと SHA-256 チェックサム：[Releases](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases)。

**GitLab CI で使う**：[`hallucination-gate`](https://gitlab.com/explore/catalog/mattbusel/llm-ci) CI/CD コンポーネントを使えば、保存した LLM の回答に信頼度の低い単語があるときにパイプラインを失敗させられます。

## 仕組み

<img alt="実際の回答に対して検出ツールがどう動くかを示すアニメーション図。ステップ 1：すべてのトークンには logprob が付いていて、p は e の logprob 乗。各トークンは赤（自信なし）からグレー（自信あり）まで色分けされる。'A' の p は 0.49、'ord' の p は 0.57。ステップ 2：トークンが単語にまとめられ、いずれかのトークンがしきい値 0.6 を下回る単語にフラグが付く。そのため Aelbert と Dordrecht にフラグが付く。ステップ 3：フラグの付いた各スパンに、モデルが検討した候補が表示される。A の代わりに The 0.43、ord の代わりに üsseldorf 0.39。" src="docs/img/how-it-works.svg" width="100%">

モデルは回答を書くとき、各トークン（単語または単語の一部）を候補のリストから選びます。候補にはそれぞれ確率があります。API はこれを `logprobs` として返します。ハルシネーションは、この確率が下がるところに潜んでいることがよくあります。人名、日付、モデルがうろ覚えの都市名などです。このツールはその数値をヒートマップにし、怪しい単語にフラグを付け、モデルが言いかけた別の候補も示します。ツール自体はモデルを持たず、`--live` を指定しない限り API も呼び出しません。

## 例

Llama 3.1 8B Instruct の実際の回答で、`examples/logprobs/` に同梱しています。`cargo run --example detect`（しきい値 0.6）の出力にある 4 つの回答のうち、最初の 2 つです。

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

Dordrecht の回答は正しいのですが、モデルは Düsseldorf に 39% の確率を与えていました。まさに確認し直すべき種類の主張です。「Aelbert」のフラグは、名前で書き始めるか「The」で書き始めるかでモデルが迷った結果です。確率が低いのは事実ではなく言い回しのせいということもあります。

1 つ目の回答のターミナル出力（`--logprobs-file examples/logprobs/cuyp.json --threshold 0.6`）：

<img alt="examples/logprobs/cuyp.json に対する llm-token-visualizer のターミナル出力。Aelbert と Dordrecht に下線が引かれた回答と、その下に弱いトークンごとの候補の棒グラフ：A 0.49、The 0.43、D 0.08、そして ord 0.57、üsseldorf 0.39、elf 0.03。" src="docs/assets/terminal-cuyp.png" width="720">

### このツールにわからないこと

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/confidently-wrong-dark.png">
  <img alt="moonwalk サンプル：'Pete Conrad was the second person to walk on the Moon'。Pete Conrad は誤りとして枠で囲まれているが、フラグは付いていない：P 0.72、ete 0.76、Conrad 1.00。代わりにフラグが付いたのは was 0.60、which 0.14、during 0.55。" src="docs/assets/confidently-wrong-light.png" width="100%">
</picture>

モデルは自信満々に間違えることがあります。`moonwalk.json` ではモデルが「Pete Conrad was the second person to walk on the Moon」と答えています（正しくは Buzz Aldrin）が、名前のどのトークンにも 72% 以上の確率が付いているので、間違った名前にフラグは付きません。このツールは回答のお墨付きを与えるものではなく、どこから確認するかを決めるために使ってください。

## このツールを選ぶ理由

logprob ベースで信頼度をチェックする Rust クレートはほかにありません。Python には [LM-Polygraph](https://github.com/IINemo/lm-polygraph) や [UQLM](https://github.com/cvs-health/uqlm) があり、はるかに多くの不確実性推定手法（サンプリングの一貫性、主張単位のスコアリング、学習済みの推定器）を備えていますが、モデルの読み込みや、1 つの回答につき複数回の生成が必要です。このツールがやるのは、保存済みのレスポンス 1 つから安くできることひとつだけです。どの単語の確率が低かったか、モデルがほかに何を検討していたかを、ターミナルのヒートマップ、HTML ページ、マージリクエスト用の Markdown、JSON、または CI の終了コードとして示します。各レポートには回答のパープレキシティ（perplexity）も、フラグの付いた各スパンにはその最も弱いトークンでの候補のエントロピーも出ます。

## 3 ステップで使う

1. **logprobs 付きのレスポンスを用意する**：API に `"logprobs": true, "top_logprobs": 3` を指定して JSON を保存します。またはツールに任せることもできます。`OPENAI_API_KEY` を設定して `llm-token-visualizer --live "your question" --save answer.json` を実行してください。
2. **実行する**：`llm-token-visualizer --logprobs-file answer.json --threshold 0.6`
3. **共有する、またはゲートにする**：送れるページを作るなら `--format html -o report.html`、PR コメントなら `--format markdown`、何かにフラグが付いたら CI ジョブを失敗させるなら `--fail-on-flag`。

## プロバイダー

`--live` はモデルに質問し、その回答を分析します。問い合わせ先は `--provider` で選び（デフォルトは `openai`）、アドレスは `--base-url` で上書きできます。

| `--provider` | デフォルトのベース URL | キーの取得元 | デフォルトの `--model` | 確認方法 |
|---|---|---|---|---|
| `openai` | `https://api.openai.com/v1`（または `OPENAI_BASE_URL`） | `OPENAI_API_KEY` | `gpt-4o-mini` | ドキュメント（`logprobs`、`top_logprobs` は最大 5）。リクエスト組み立てはユニットテスト済み。 |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | ドキュメント（`logprobs`、`top_logprobs`。対応状況は上流のプロバイダー次第なので、リクエストで `provider.require_parameters` を設定）。リクエスト組み立てはユニットテスト済み。 |
| `together` | `https://api.together.ai/v1` | `TOGETHER_API_KEY` | なし、`--model` を指定 | ドキュメント（Together は `true` ではなく `"logprobs": <int>` を受け付けるので、CLI はそれを送る）。リクエスト組み立てはユニットテスト済み。 |
| `vllm` | `http://localhost:8000/v1` | `VLLM_API_KEY`、サーバーが `--api-key` を使っている場合のみ | なし、`--model` を指定 | ドキュメント（Chat Completions の `logprobs`、`top_logprobs`）。リクエスト組み立てはユニットテスト済み。 |
| `ollama` | `http://localhost:11434/v1` | キー不要 | なし、`--model` を指定 | `qwen2.5-coder:14b` で Ollama 0.34.4 に**実際に呼び出し**：logprobs と上位 3 つの候補が返ってきた。（Ollama 自身の OpenAI 互換ページではまだ logprobs が未対応と書かれており、古いバージョンでは返らない可能性がある。） |

```sh
llm-token-visualizer --live "Who painted The Night Watch?" --provider ollama --model qwen2.5-coder:14b
llm-token-visualizer --live "Who painted The Night Watch?" --provider openrouter --save answer.json
llm-token-visualizer --live "..." --provider vllm --model my-model --base-url http://gpu-box:8000/v1
```

このリリースでは、ホスト型のプロバイダーを実際のキーで呼び出してはいません。プロバイダーやモデルが logprobs なしで回答した場合、CLI は空のレポートを出す代わりに、その旨を伝えるエラーで停止します（回答は表示します）。Anthropic の API は logprobs を返さないので、プリセットはありません。

Google Gemini のネイティブ API も logprobs を返します（`generationConfig: {responseLogprobs: true, logprobs: 3}`）。レスポンスを保存して `--logprobs-file` で渡してください。

API キーが手元にない？まずはサンプルをどうぞ：`curl -LO https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/raw/main/examples/logprobs/cuyp.json`

## ライブラリ

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

| フィーチャー | デフォルト | 追加されるもの |
|---|---|---|
| `live` | オン | `--live` と `live` モジュール：OpenAI 互換 API に問い合わせる（ureq、rustls、タイムアウト 120 秒） |
| `async-openai` | オフ | `async_openai` の chat completion レスポンス向けの `interop::async_openai::tokens_from_response`（型のみ、HTTP クライアントなし） |

サンプル（`cargo run --example <name>`）：`detect`（同梱のすべてのサンプル）、`ci_gate`（回答のディレクトリを処理し、どれかにフラグが付けば終了コード 2）、`html_report`（1 つの回答の HTML ページを書き出す）。

すでに **async-openai** を使っていますか？このフィーチャーを有効にして、`CreateChatCompletionResponse` を `interop::async_openai::tokens_from_response` に渡し、`detect` を呼んでください。別のクライアントを使っているなら、レスポンスを JSON にシリアライズして `parse_logprobs` を呼びます。

## パフォーマンス

`cargo bench --bench vs_0_4` は 4,000 トークンのレスポンスをパースしてスパンにフラグを付けます。0.5.0 では 9.0 ms、0.4.0 では 14.6 ms でした（criterion の中央値、i7-13700KF、Windows 11、Rust 1.91）。0.5.0 は JSON を先にコピーせずに logprobs をデシリアライズするので、Unicode の単語分割のコストを差し引いてもおつりが来ます。

## ドキュメント

| | |
|---|---|
| [リファレンス](docs/REFERENCE.md) | すべてのフラグ、出力形式（ターミナル、HTML、Markdown、JSON）、CI での使い方、入力形式、ライブモード、独自の信頼度スコア、ライブラリ API |
| [仕組みとリポジトリ構成](docs/ARCHITECTURE.md) | 検出ルールと配色の詳細、ソースの構成、どこが試作段階でどこが出荷済みか |
| [docs.rs の API ドキュメント](https://docs.rs/llm-token-visualizer) | Rust ライブラリ |
| [プロジェクトサイト](https://hallucination-highlighter.vercel.app/) | 概要 |
| [ブラウザ版（ソース）](docs/try/index.html) | しきい値スライダー付きのサンプル、または自分のキーでプロバイダーに質問。クローンしたリポジトリからファイルを開いてください |
| [変更履歴](CHANGELOG.md) | 各リリースでの変更点 |

## ライセンス

MIT。[LICENSE](LICENSE) を参照してください。
