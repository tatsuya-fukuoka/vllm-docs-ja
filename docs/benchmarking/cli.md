# Benchmark CLI

この節では、vLLM がサポートする豊富なデータセットを使ってベンチマークを実行する方法を説明します。

これは生きたドキュメントであり、新しい機能やデータセットが利用可能になるたびに更新されます。

!!! tip
    このページで説明するベンチマークは、主に vLLM の特定機能の評価とリグレッションテストを目的としています。

    本番の vLLM サーバーのベンチマークには [GuideLLM](https://github.com/vllm-project/guidellm) を推奨します。これは実績のある性能ベンチマークのフレームワークで、進捗のライブ表示とレポートの自動生成に対応しています。また、データセットの読み込み、リクエストの整形、ワークロードのパターンという点で `vllm bench serve` より柔軟です。

## データセットの概要 { #dataset-overview }

<style>
th {
  min-width: 0 !important;
}
</style>

| データセット | オンライン | オフライン | データの取得元 |
| ------- | ------ | ------- | --------- |
| ShareGPT | ✅ | ✅ | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| ShareGPT4V (Image) | ✅ | ✅ | `wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json`<br>Note that the images need to be downloaded separately. For example, to download COCO's 2017 Train images:<br>`wget http://images.cocodataset.org/zips/train2017.zip` |
| ShareGPT4Video (Video) | ✅ | ✅ | `git clone https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video` |
| BurstGPT | ✅ | ✅ | `wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv` |
| Sonnet (deprecated) | ✅ | ✅ | Local file: `benchmarks/sonnet.txt` |
| Random | ✅ | ✅ | `synthetic` |
| RandomMultiModal (Image/Video) | ✅ | ✅ | `synthetic` |
| RandomForReranking | ✅ | ✅ | `synthetic` |
| Prefix Repetition | ✅ | ✅ | `synthetic` |
| HuggingFace-VisionArena | ✅ | ✅ | `lmarena-ai/VisionArena-Chat` |
| HuggingFace-MMVU | ✅ | ✅ | `yale-nlp/MMVU` |
| HuggingFace-InstructCoder | ✅ | ✅ | `likaixin/InstructCoder` |
| HuggingFace-AIMO | ✅ | ✅ | `AI-MO/aimo-validation-aime`, `AI-MO/NuminaMath-1.5`, `AI-MO/NuminaMath-CoT` |
| HuggingFace-Other | ✅ | ✅ | `lmms-lab/LLaVA-OneVision-Data`, `Aeala/ShareGPT_Vicuna_unfiltered` |
| HuggingFace-MTBench | ✅ | ✅ | `philschmid/mt-bench` |
| HuggingFace-HumanEval | ✅ | ✅ | `openai/openai_humaneval` |
| HuggingFace-GSM8K | ✅ | ✅ | `openai/gsm8k` |
| HuggingFace-Blazedit | ✅ | ✅ | `vdaita/edit_5k_char`, `vdaita/edit_10k_char` |
| HuggingFace-ASR | ✅ | ✅ | `openslr/librispeech_asr`, `facebook/voxpopuli`, `LIUM/tedlium`, `edinburghcstr/ami`, `speechcolab/gigaspeech`, `kensho/spgispeech`, `ArtificialAnalysis/Earnings22-Cleaned-AA`, `D4nt3/esb-datasets-earnings22-validation-tiny-filtered` |
| Spec Bench | ✅ | ✅ | `wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl` |
| SPEED-Bench | ✅ | ✅ | `curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \| python3 -` |
| Custom | ✅ | ✅ | Local file: `data.jsonl` |
| Custom Audio | ✅ | ✅ | Local file: `audio_data.jsonl` |
| Custom Image | ✅ | ✅ | Local file: `image_data.jsonl` |

凡例:

- ✅ - サポート済み
- 🟡 - 部分的にサポート
- 🚧 - 今後サポート予定

!!! note
    HuggingFace のデータセットでは `dataset-name` に `hf` を設定してください。
    ローカルの `dataset-path` を使う場合は、次のように `hf-name` に Hugging Face の ID を設定してください。

    ```bash
    --dataset-path /datasets/VisionArena-Chat/ --hf-name lmarena-ai/VisionArena-Chat
    ```

## 例 { #examples }

### 🚀 オンラインベンチマーク { #-online-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

まず、モデルのサービングを開始します。

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

次に、ベンチマークのスクリプトを実行します。

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 10
```

成功すると、次のような出力が表示されます。

```text
============ Serving Benchmark Result ============
Successful requests:                     10
Benchmark duration (s):                  5.78
Total input tokens:                      1369
Total generated tokens:                  2212
Request throughput (req/s):              1.73
Output token throughput (tok/s):         382.89
Total token throughput (tok/s):          619.85
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```

#### 結果の可視化 { #results-visualization }

`--plot-timeline` と `--plot-dataset-stats` を使うと、それぞれリクエスト完了のタイムラインと、データセットのプロンプト / 出力トークンの統計を生成できます。デバッグ用途やより詳しい分析に役立ちます。

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --plot-timeline \
    --timeline-itl-thresholds 2,5 \
    --plot-dataset-stats \
    --save-result
```

##### インタラクティブなタイムライン { #interactive-timeline }

生成されるタイムラインは HTML ファイル形式のインタラクティブな可視化で、ほとんどのブラウザで表示できます。ITL の色の閾値をカスタマイズするには、`--timeline-itl-thresholds` フラグ（既定: 25ms、50ms）を使います。

出力例:

<iframe src="https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/contributing/vllm_bench_serve_timeline.html" width="100%" height="600" frameborder="0"></iframe>

##### データセットの統計 { #dataset-statistics }

生成される図は、入力プロンプトと出力トークンの分布を示します。

出力例: ![データセットの統計](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/contributing/vllm_bench_serve_dataset_stats.png)

#### カスタムデータセット { #custom-dataset }

ベンチマークしたいデータセットが vLLM でまだサポートされていない場合でも、`CustomDataset` を使ってベンチマークできます。推論時にはオプション `--dataset-name custom` を指定します。データは `.jsonl` 形式で、エントリごとに "prompt" フィールドを持つ必要があります（例: data.jsonl）。

```json
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}
{"prompt": "What is the capital of China?"}
```

```bash
# start server
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```bash
# run benchmarking script
vllm bench serve --port 9001 --save-result --save-detailed \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/completions \
  --dataset-name custom \
  --dataset-path <path-to-your-data-jsonl> \
  --custom-skip-chat-template \
  --num-prompts 80 \
  --max-concurrency 1 \
  --temperature=0.3 \
  --top-p=0.75 \
  --result-dir "./log/"
```

データにすでにチャットテンプレートが適用されている場合は、`--custom-skip-chat-template` を使ってテンプレートの適用をスキップできます。

#### カスタム音声データセット { #custom-audio-dataset }

ベンチマークしたい音声データセットが vLLM でまだサポートされていない場合は、`CustomAudioDataset` を使ってベンチマークできます。推論時にはオプション `--dataset-name custom_audio` を指定します。データは `.jsonl` 形式で、エントリごとに "prompt" と "audio" のフィールドを持つ必要があります（例: `audio_data.jsonl`）。

```json
{"prompt": "What does this audio say?", "audio": "/path/to/audio_1.wav"}
{"prompt": "Transcribe the audio.", "audio": "/path/to/audio_2.wav"}
```

- **Supported models:** The `CustomAudioDataset` class supports two types of audio models: ASR models (e.g. Whisper) which do not require a "prompt" field; and multimodal audio-text chat models (e.g. Qwen2-Audio). Since these model types require different arguments at inference, we are giving two examples.

- **Example 1: Whisper**

Whisper は音声認識専用のエンコーダ・デコーダモデルであるため、`--backend openai-audio` と `--endpoint /v1/audio/transcriptions` を使います。

```bash
# start server
vllm serve openai/whisper-tiny
```

```bash
vllm bench serve \
  --model openai/whisper-tiny \
  --backend openai-audio \
  --endpoint /v1/audio/transcriptions \
  --dataset-name custom_audio \
  --dataset-path audio_data.jsonl \
  --no-oversample \
  --custom-output-len 256 \
  --save-result \
  --save-detailed \
  --result-filename whisper_bench.json
```

- **Example 2: Qwen2-Audio**

Qwen2-Audio は音声認識と音声解析ができるマルチモーダルのチャットモデルであるため、`--backend openai-chat` と `--endpoint /v1/chat/completions` を使います。また、マルチモーダルのチャット変換を有効にするために `--enable-multimodal-chat` も必要です。

```bash
vllm bench serve \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name custom_audio \
  --dataset-path audio_data.jsonl \
  --no-oversample \
  --custom-output-len 256 \
  --enable-multimodal-chat \
  --save-result \
  --save-detailed \
  --result-filename qwen_bench.json
```

#### カスタム画像データセット { #custom-image-dataset }

ベンチマークしたい画像データセットが vLLM でまだサポートされていない場合は、`CustomImageDataset` を使ってベンチマークできます。推論時にはオプション `--dataset-name custom_image` を指定します。データは `.jsonl` 形式で、エントリごとに "prompt" と "image_files" のフィールドを使えます（例: `image_data.jsonl`）。

```json
{"prompt": "How many animals are present in the given image?", "image_files": ["/path/to/image/folder/horsepony.jpg"]}
{"prompt": "What colour is the bird shown in the image?", "image_files": ["/path/to/image/folder/flycatcher.jpeg"]}
```

"image_files" に列挙した画像は、プロンプトのテキストの後ろに列挙順でリクエストへ追加されます。テキストと画像を交互に並べた順序を保つには、OpenAI 互換のコンテンツパートを持つ "content" フィールドを使います。

```json
{"content": [{"type": "text", "text": "Compare "}, {"type": "image", "image": "/path/to/image/folder/chart_a.png"}, {"type": "text", "text": " with "}, {"type": "image_url", "image_url": {"url": "/path/to/image/folder/chart_b.png"}}]}
```

"image" の省略形は "image_files" と同じ値を受け付けます。"image_url" フィールドは、"url" フィールドを持つ OpenAI 形式のオブジェクトか、URL の文字列のいずれかを受け付けます。

既定では、画像への参照はそのままサービングのエンドポイントへ送られ、ローカルの画像パスは `file://` の URL に変換されます。

ベンチマークのクライアントがリクエスト送信前にローカルおよび HTTP(S) の画像を読み込むようにするには、`--custom-ensure-client-side-data` を渡してクライアント側で base64 のデータ URL としてエンコードします。

既存の `data:image/...` の URL はすでに自己完結しているため、そのまま維持されます。

```bash
# need a model with vision capability here
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
# run benchmarking script
vllm bench serve --save-result --save-detailed \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom_image \
  --dataset-path <path-to-your-image-data-jsonl> \
  --custom-ensure-client-side-data
```

マルチモーダル入力では、`openai-chat` バックエンドと `/v1/chat/completions` エンドポイントを使う必要がある点に注意してください。

#### vision-language モデル向けの VisionArena ベンチマーク { #visionarena-benchmark-for-vision-language-models }

```bash
# need a model with vision capability here
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 1000
```

#### 投機的デコーディングを用いた InstructCoder のベンチマーク { #instructcoder-benchmark-with-speculative-decoding }

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name hf \
    --dataset-path likaixin/InstructCoder \
    --num-prompts 2048
```

#### 投機的デコーディングを用いた Spec Bench のベンチマーク { #spec-bench-benchmark-with-speculative-decoding }

``` bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

[SpecBench dataset](https://github.com/hemingkx/Spec-Bench)

すべてのカテゴリを実行します。

``` bash
# Download the dataset using:
# wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl

vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/spec_bench/question.jsonl" \
    --num-prompts -1
```

利用できるカテゴリは `[writing, roleplay, reasoning, math, coding, extraction, stem, humanities, translation, summarization, qa, math_reasoning, rag]` です。

「summarization」のような特定のカテゴリのみを実行します。

``` bash
vllm bench serve \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name spec_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/spec_bench/question.jsonl" \
    --num-prompts -1 \
    --spec-bench-category "summarization"
```

#### 投機的デコーディングを用いた SPEED-Bench のベンチマーク { #speed-bench-benchmark-with-speculative-decoding }

[SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) is a unified and diverse dataset for speculative decoding, supporting acceptance rate and length measurements using the Qualitative split and throughput measurements using the Throughput splits in 5 configuration of input sequence length (1k, 2k, 8k, 16k, 32k).

!!! note
    This dataset is governed by the [NVIDIA Evaluation Dataset License Agreement](https://huggingface.co/datasets/nvidia/SPEED-Bench/blob/main/License.pdf). For each dataset a user elects to use, the user is responsible for checking if the dataset license is fit for the intended purpose. The `prepare.py` script automatically fetches data from all the source datasets.

まず、次のワンライナーでデータセットをフォルダへダウンロードします。

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py | python3 -
```

このコマンドは次の引数もサポートします。

- `--config`: download only a subset of the dataset: `qualitative`, `throughput_1k`, `throughput_2k`, `throughput_8k`, `throughput_16k` and `throughput_32k`. By default, it will download all subsets.
- `--output_dir`: download to a specified folder. By default, it will download to the current directory.

投機的デコーディングを有効にしてサーバーを起動します。

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --speculative-config $'{"method": "eagle3",
    "num_speculative_tokens": 3,
    "model": "nvidia/Llama-3.3-70B-Instruct-Eagle3"}'
```

Qualitative スプリットのすべてのカテゴリを実行します。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name speed_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/speed_bench" \
    --num-prompts -1
```

利用できるカテゴリは `[writing, roleplay, reasoning, math, coding, stem, humanities, multilingual, summarization, qa, rag]` です。

「multilingual」のような特定のカテゴリのみを実行します。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name speed_bench \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/speed_bench" \
    --num-prompts -1 \
    --speed-bench-category "multilingual"
```

Throughput スプリット（2k ISL）のすべてのカテゴリを実行します。

```bash
vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name speed_bench \
    --speed-bench-dataset-subset throughput_2k \
    --dataset-path "<YOUR_DOWNLOADED_PATH>/data/speed_bench/" \
    --num-prompts -1
```

利用できるカテゴリは `[high_entropy, mixed, low_entropy]` です。high entropy のデータは創作文のような非構造的なデータを、low entropy のデータはコーディングのようなより構造的なデータを含みます。詳細はデータセットカードを参照してください。

#### BFCL（ツール呼び出し）のベンチマーク { #bfcl-tool-calling-benchmark }

Berkeley Function Calling Leaderboard（BFCL）のデータセットは、現実的なツール呼び出しのトラフィックにおける
サービングのレイテンシとスループットを測定します。各リクエストはサンプルごとの `tools` スキーマと
チャット履歴を持つため、サーバーは自動ツール選択のパーサーを有効にした `/v1/chat/completions` を
公開している必要があります。ベンチマークのクライアントは常に `openai-chat` バックエンドを使います。

ツールパーサーを有効にしたサーバーを起動してからベンチマークを実行します。たとえば `gpt-oss-20b` の場合は次のようにします。

```bash
# Server
vllm serve openai/gpt-oss-20b \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --reasoning-parser openai_gptoss

# Client
vllm bench serve \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --model openai/gpt-oss-20b \
    --dataset-name hf \
    --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --bfcl-categories simple,live_simple,multiple \
    --num-prompts 200
```

`--bfcl-categories` は BFCL v3 のカテゴリ名（`BFCL_v3_` の接頭辞と `.json` の拡張子を除いたもの）を
カンマ区切りで並べたものです。既定値は `simple,live_simple,multiple` です。マルチターン以外で
サポートされる他のカテゴリには `parallel`、`live_parallel`、`parallel_multiple`、
`live_parallel_multiple`、`irrelevance`、`live_irrelevance`、`live_relevance`、`java`、
`javascript`、`rest` があります。マルチターンのカテゴリはまだサポートされていません。

このデータセットクラスは BFCL の緩いスキーマ方言を正規化するため（`dict` → `object`、
`float` → `number`、`tuple` → `array`、`any` → `string`）、最近の文法バックエンドでも
変換後のツール定義を受け付けられます。

#### その他の HuggingFaceDataset の例 { #other-huggingfacedataset-examples }

```bash
vllm serve Qwen/Qwen2-VL-7B-Instruct
```

`lmms-lab/LLaVA-OneVision-Data`:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path AI-MO/aimo-validation-aime \
    --num-prompts 10 \
    --seed 42
```

`philschmid/mt-bench`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80
```

`openai/openai_humaneval`:

``` bash
vllm bench serve \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --dataset-name hf \
    --dataset-path openai/openai_humaneval \
    --num-prompts 80
```

`openai/gsm8k`:

``` bash
vllm bench serve \
    --model NousResearch/Hermes-3-Llama-3.1-8B \
    --dataset-name hf \
    --dataset-path openai/gsm8k \
    --num-prompts 80
```

`vdaita/edit_5k_char` or `vdaita/edit_10k_char`:

``` bash
vllm bench serve \
    --model Qwen/QwQ-32B \
    --dataset-name hf \
    --dataset-path vdaita/edit_5k_char \
    --num-prompts 90 \
    --blazedit-min-distance 0.01 \
    --blazedit-max-distance 0.99
```

`openslr/librispeech_asr`, `facebook/voxpopuli`, `LIUM/tedlium`, `edinburghcstr/ami`, `speechcolab/gigaspeech`, `kensho/spgispeech`, `ArtificialAnalysis/Earnings22-Cleaned-AA`, `D4nt3/esb-datasets-earnings22-validation-tiny-filtered`

```bash
vllm bench serve \
    --model openai/whisper-large-v3-turbo \
    --backend openai-audio \
    --dataset-name hf \
    --dataset-path facebook/voxpopuli --hf-subset en --hf-split test --no-stream --trust-remote-code \
    --num-prompts 99999999 \
    --no-oversample \
    --endpoint /v1/audio/transcriptions \
    --ready-check-timeout-sec 600 \
    --save-result \
    --max-concurrency 512
```

#### サンプリングパラメータを指定して実行する { #running-with-sampling-parameters }

`vllm` のような OpenAI 互換バックエンドを使う場合、任意のサンプリングパラメータを指定できます。
クライアント側のコマンド例:

```bash
vllm bench serve \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --top-k 10 \
  --top-p 0.9 \
  --temperature 0.5 \
  --num-prompts 10
```

#### リクエストレートをランプアップして実行する { #running-with-ramp-up-request-rate }

このベンチマークツールは、実行時間の経過とともにリクエストレートを増加させる（ランプアップ）ことも
サポートします。サーバーのストレステストや、あるレイテンシ予算のもとで処理できる最大スループットを
見つけるのに役立ちます。

2 つのランプアップ戦略がサポートされています。

- `linear`: Increases the request rate linearly from a start value to an end value.
- `exponential`: Increases the request rate exponentially.

ランプアップの制御には、次の引数を使えます。

- `--ramp-up-strategy`: The ramp-up strategy to use (`linear` or `exponential`).
- `--ramp-up-start-rps`: The request rate at the beginning of the benchmark.
- `--ramp-up-end-rps`: The request rate at the end of the benchmark.

#### 負荷パターンの設定 { #load-pattern-configuration }

vLLM's benchmark serving script provides sophisticated load pattern simulation capabilities through three key parameters that control request generation and concurrency behavior:

##### Load Pattern Control Parameters

- `--request-rate`: Controls the target request generation rate (requests per second). Set to `inf` for maximum throughput testing or finite values for controlled load simulation.
- `--burstiness`: Controls traffic variability using a Gamma distribution (range: > 0). Lower values create bursty traffic, higher values create uniform traffic.
- `--max-concurrency`: Limits concurrent outstanding requests. If this argument is not provided, concurrency is unlimited. Set a value to simulate backpressure.

These parameters work together to create realistic load patterns with carefully chosen defaults. The `--request-rate` parameter defaults to `inf` (infinite), which sends all requests immediately for maximum throughput testing. When set to finite values, it uses either a Poisson process (default `--burstiness=1.0`) or Gamma distribution for realistic request timing. The `--burstiness` parameter only takes effect when `--request-rate` is not infinite - a value of 1.0 creates natural Poisson traffic, while lower values (0.1-0.5) create bursty patterns and higher values (2.0-5.0) create uniform spacing. The `--max-concurrency` parameter defaults to `None` (unlimited) but can be set to simulate real-world constraints where a load balancer or API gateway limits concurrent connections. When combined, these parameters allow you to simulate everything from unrestricted stress testing (`--request-rate=inf`) to production-like scenarios with realistic arrival patterns and resource constraints.

The `--burstiness` parameter mathematically controls request arrival patterns using a Gamma distribution where:

- Shape parameter: `burstiness` value
- Coefficient of Variation (CV): $\frac{1}{\sqrt{burstiness}}$
- Traffic characteristics:
    - `burstiness = 0.1`: Highly bursty traffic (CV ≈ 3.16) - stress testing
    - `burstiness = 1.0`: Natural Poisson traffic (CV = 1.0) - realistic simulation  
    - `burstiness = 5.0`: Uniform traffic (CV ≈ 0.45) - controlled load testing

![Load Pattern Examples](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/contributing/load-pattern-examples.png)

*Figure: Load pattern examples for each use case. Top row: Request arrival timelines showing cumulative requests over time. Bottom row: Inter-arrival time distributions showing traffic variability patterns. Each column represents a different use case with its specific parameter settings and resulting traffic characteristics.*

Load Pattern Recommendations by Use Case:

| Use Case           | Burstiness   | Request Rate    | Max Concurrency | Description                                                                        |
| ---                | ---          | ---             | ---             | ---                                                                                |
| Maximum Throughput | N/A          | Infinite        | Limited         | **Most common**: Simulates load balancer/gateway limits with unlimited user demand |
| Realistic Testing  | 1.0          | Moderate (5-20) | Infinite        | Natural Poisson traffic patterns for baseline performance                          |
| Stress Testing     | 0.1-0.5      | High (20-100)   | Infinite        | Challenging burst patterns to test resilience                                      |
| Latency Profiling  | 2.0-5.0      | Low (1-10)      | Infinite        | Uniform load for consistent timing analysis                                        |
| Capacity Planning  | 1.0          | Variable        | Limited         | Test resource limits with realistic constraints                                    |
| SLA Validation     | 1.0          | Target rate     | SLA limit       | Production-like constraints for compliance testing                                 |

These load patterns help evaluate different aspects of your vLLM deployment, from basic performance characteristics to resilience under challenging traffic conditions.

The **Maximum Throughput** pattern (`--request-rate=inf --max-concurrency=<limit>`) is the most commonly used configuration for production benchmarking. This simulates real-world deployment architectures where:

- Users send requests as fast as they can (infinite rate)
- A load balancer or API gateway controls the maximum concurrent connections
- The system operates at its concurrency limit, revealing true throughput capacity
- `--burstiness` has no effect since request timing is not controlled when rate is infinite

This pattern helps determine optimal concurrency settings for your production load balancer configuration.

To effectively configure load patterns, especially for **Capacity Planning** and **SLA Validation** use cases, you need to understand your system's resource limits. During startup, vLLM reports KV cache configuration that directly impacts your load testing parameters:

```text
GPU KV cache size: 15,728,640 tokens
Maximum concurrency for 8,192 tokens per request: 1920
```

Where:

- GPU KV cache size: Total tokens that can be cached across all concurrent requests
- Maximum concurrency: Theoretical maximum concurrent requests for the given `max_model_len`
- Calculation: `max_concurrency = kv_cache_size / max_model_len`

Using KV cache metrics for load pattern configuration:

- For Capacity Planning: Set `--max-concurrency` to 80-90% of the reported maximum to test realistic resource constraints
- For SLA Validation: Use the reported maximum as your SLA limit to ensure compliance testing matches production capacity
- For Realistic Testing: Monitor memory usage when approaching theoretical limits to understand sustainable request rates
- Request rate guidance: Use the KV cache size to estimate sustainable request rates for your specific workload and sequence lengths

</details>

### 📈 オフラインのスループットベンチマーク { #-offline-throughput-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

```bash
vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path vllm/benchmarks/sonnet.txt \
  --num-prompts 10
```

If successful, you will see the following output

```text
Throughput: 7.15 requests/s, 4656.00 total tokens/s, 1072.15 output tokens/s
Total num prompt tokens:  5014
Total num output tokens:  1500
```

#### vision-language モデル向けの VisionArena ベンチマーク { #visionarena-benchmark-for-vision-language-models_1 }

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 1000 \
  --hf-split train
```

The `num prompt tokens` now includes image token counts

```text
Throughput: 2.55 requests/s, 4036.92 total tokens/s, 326.90 output tokens/s
Total num prompt tokens:  14527
Total num output tokens:  1280
```

#### 投機的デコーディングを用いた InstructCoder のベンチマーク { #instructcoder-benchmark-with-speculative-decoding_1 }

``` bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm bench throughput \
    --dataset-name=hf \
    --dataset-path=likaixin/InstructCoder \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --input-len=1000 \
    --output-len=100 \
    --num-prompts=2048 \
    --async-engine \
    --speculative-config $'{"method": "ngram",
    "num_speculative_tokens": 5, "prompt_lookup_max": 5,
    "prompt_lookup_min": 2}'
```

```text
Throughput: 104.77 requests/s, 23836.22 total tokens/s, 10477.10 output tokens/s
Total num prompt tokens:  261136
Total num output tokens:  204800
```

#### その他の HuggingFaceDataset の例 { #other-huggingfacedataset-examples_1 }

`lmms-lab/LLaVA-OneVision-Data`:

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path lmms-lab/LLaVA-OneVision-Data \
  --hf-split train \
  --hf-subset "chart2text(cauldron)" \
  --num-prompts 10
```

`Aeala/ShareGPT_Vicuna_unfiltered`:

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name hf \
  --dataset-path Aeala/ShareGPT_Vicuna_unfiltered \
  --hf-split train \
  --num-prompts 10
```

`AI-MO/aimo-validation-aime`:

```bash
vllm bench throughput \
  --model Qwen/QwQ-32B \
  --backend vllm \
  --dataset-name hf \
  --dataset-path AI-MO/aimo-validation-aime \
  --hf-split train \
  --num-prompts 10
```

Benchmark with LoRA adapters:

``` bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench throughput \
  --model meta-llama/Llama-2-7b-hf \
  --backend vllm \
  --dataset_path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset_name sharegpt \
  --num-prompts 10 \
  --max-loras 2 \
  --max-lora-rank 8 \
  --enable-lora \
  --lora-path yard1/llama-2-7b-sql-lora-test
```

#### 合成のランダムなマルチモーダル入力（random-mm） { #synthetic-random-multimodal-random-mm }

Generate synthetic multimodal inputs for offline throughput testing without external datasets.
Use `--backend vllm-chat` so that image tokens are counted correctly.

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --backend vllm-chat \
  --dataset-name random-mm \
  --num-prompts 100 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

</details>

### 🛠️ 構造化出力のベンチマーク { #-structured-output-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of structured output generation (JSON, grammar, regex).

#### サーバーのセットアップ { #server-setup }

```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-8B
```

#### JSON スキーマのベンチマーク { #json-schema-benchmark }

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset json \
  --structured-output-ratio 1.0 \
  --request-rate 10 \
  --num-prompts 1000
```

#### 文法にもとづく生成のベンチマーク { #grammar-based-generation-benchmark }

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset grammar \
  --structure-type grammar \
  --request-rate 10 \
  --num-prompts 1000
```

#### 正規表現にもとづく生成のベンチマーク { #regex-based-generation-benchmark }

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset regex \
  --request-rate 10 \
  --num-prompts 1000
```

#### 選択肢にもとづく生成のベンチマーク { #choice-based-generation-benchmark }

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset choice \
  --request-rate 10 \
  --num-prompts 1000
```

#### XGrammar のベンチマークデータセット { #xgrammar-benchmark-dataset }

```bash
python3 benchmarks/benchmark_serving_structured_output.py \
  --backend vllm \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset xgrammar_bench \
  --request-rate 10 \
  --num-prompts 1000
```

</details>

### 📚 長文ドキュメント QA のベンチマーク { #-long-document-qa-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of long document question-answering with prefix caching.

#### 基本的な長文ドキュメント QA のテスト { #basic-long-document-qa-test }

```bash
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 16 \
  --document-length 2000 \
  --output-len 50 \
  --repeat-count 5
```

#### 各種の繰り返しモード { #different-repeat-modes }

```bash
# Random mode (default) - shuffle prompts randomly
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode random

# Tile mode - repeat entire prompt list in sequence
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode tile

# Interleave mode - repeat each prompt consecutively
python3 benchmarks/benchmark_long_document_qa_throughput.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-documents 8 \
  --document-length 3000 \
  --repeat-count 3 \
  --repeat-mode interleave
```

</details>

### 🗂️ プレフィックスキャッシュのベンチマーク { #-prefix-caching-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the efficiency of automatic prefix caching.

#### 固定プロンプトでのプレフィックスキャッシュ { #fixed-prompt-with-prefix-caching }

```bash
python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --enable-prefix-caching \
  --num-prompts 1 \
  --repeat-count 100 \
  --input-length-range 128:256
```

#### ShareGPT データセットでのプレフィックスキャッシュ { #sharegpt-dataset-with-prefix-caching }

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 benchmarks/benchmark_prefix_caching.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-path /path/ShareGPT_V3_unfiltered_cleaned_split.json \
  --enable-prefix-caching \
  --num-prompts 20 \
  --repeat-count 5 \
  --input-length-range 128:256
```

##### プレフィックス反復のデータセット { #prefix-repetition-dataset }

```bash
vllm bench serve \
  --backend openai \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-name prefix_repetition \
  --num-prompts 100 \
  --prefix-repetition-prefix-len 512 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 5 \
  --prefix-repetition-output-len 128
```

</details>

### タイミング付きトレースの再生 { #replay-timed-traces }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Example of how to run traces which have timing information
with them.

#### MoonshotAI のトレースを実行する { #running-moonshotai-traces }

サーバーを起動します。

```bash
vllm serve Qwen/Qwen3.5-2B \
--host 127.0.0.1 --port 8000
```

ベンチマークを実行します。

```bash
# Download an example trace 
# curl -L -o conversation_trace.jsonl \
#https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl 

vllm bench serve --model Qwen/Qwen3.5-2B \  
--dataset-name=timed_trace --num-prompts 100 --host 127.0.0.1 \
--port 8000 --dataset-path ./conversation_trace.jsonl \
--ignore-eos  --self-timed --timed-trace-chunk-hash-size 512 \
--timed-trace-sec-multiplier 0.001 
```

This will replay the first 100 lines from the trace file `conversation.jsonl`.  

</details>

### 🧪 ハッシュのベンチマーク { #-hashing-benchmarks }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Two helper scripts live in `benchmarks/` to compare hashing options used by prefix caching and related utilities. They are standalone (no server required) and help choose a hash algorithm before enabling prefix caching in production.

- `benchmarks/benchmark_hash.py`: Micro-benchmark that measures per-call latency of three implementations on a representative `(bytes, tuple[int])` payload.

```bash
python benchmarks/benchmark_hash.py --iterations 20000 --seed 42
```

- `benchmarks/benchmark_prefix_block_hash.py`: End-to-end block hashing benchmark that runs the full prefix-cache hash pipeline (`hash_block_tokens`) across many fake blocks and reports throughput.

```bash
python benchmarks/benchmark_prefix_block_hash.py --num-blocks 20000 --block-size 32 --trials 5
```

Supported algorithms: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`. Install optional deps to exercise all variants:

```bash
uv pip install xxhash cbor2
```

If an algorithm’s dependency is missing, the script will skip it and continue.

</details>

### ⚡ リクエスト優先度付けのベンチマーク { #-request-prioritization-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of request prioritization in vLLM.

#### 基本的な優先度付けのテスト { #basic-prioritization-test }

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority
```

#### プロンプトあたり複数シーケンス { #multiple-sequences-per-prompt }

```bash
python3 benchmarks/benchmark_prioritization.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --input-len 128 \
  --output-len 64 \
  --num-prompts 100 \
  --scheduling-policy priority \
  --n 2
```

</details>

### 👁️ マルチモーダルのベンチマーク { #-multi-modal-benchmark }

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

Benchmark the performance of multi-modal requests in vLLM.

#### 画像（ShareGPT4V） { #images-sharegpt4v }

vLLM を起動します。

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path /path/to/sharegpt4v/images
```

Send requests with images:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions
```

#### 動画（ShareGPT4Video） { #videos-sharegpt4video }

vLLM を起動します。

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"video": 1}' \
  --allowed-local-media-path /path/to/sharegpt4video/videos
```

Send requests with videos:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT4Video/llava_v1_5_mix665k_with_video_chatgpt72k_share4video28k.json \
  --num-prompts 100 \
  --save-result \
  --result-dir ~/vllm_benchmark_results \
  --save-detailed \
  --endpoint /v1/chat/completions
```

#### 合成のランダムな画像（random-mm） { #synthetic-random-images-random-mm }

Generate synthetic image inputs alongside random text prompts to stress-test vision models without external datasets.

Notes:

- For online benchmarks, use `--backend openai-chat` with endpoint `/v1/chat/completions`.
- For offline benchmarks, use `--backend vllm-chat` (see [Offline Throughput Benchmark](#-offline-throughput-benchmark) for an example).

Start the server (example):

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --mm-processor-kwargs max_pixels=1003520
```

Benchmark. It is recommended to use the flag `--ignore-eos` to simulate real responses. You can set the size of the output via the arg `random-output-len`.

Ex.1: Fixed number of items and a single image resolution, enforcing generation of approx 40 tokens:

```bash
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random-mm \
  --num-prompts 100 \
  --max-concurrency 10 \
  --random-prefix-len 25 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-range-ratio 0.2 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(224, 224, 1): 1.0}' \
  --request-rate inf \
  --ignore-eos \
  --seed 42
```

The number of items per request can be controlled by passing multiple image buckets:

```bash
  --random-mm-base-items-per-request 2 \
  --random-mm-num-mm-items-range-ratio 0.5 \
  --random-mm-limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}' \
```

Flags specific to `random-mm`:

- `--random-mm-base-items-per-request`: base number of multimodal items per request.
- `--random-mm-num-mm-items-range-ratio`: vary item count uniformly in the closed integer range [floor(n·(1−r)), ceil(n·(1+r))]. Set r=0 to keep it fixed; r=1 allows 0 items.
- `--random-mm-limit-mm-per-prompt`: per-modality hard caps, e.g. '{"image": 3, "video": 0}'.
- `--random-mm-bucket-config`: dict mapping (H, W, T) → probability. Entries with probability 0 are removed; remaining probabilities are renormalized to sum to 1. Use T=1 for images. Set any T>1 for videos (video sampling not yet supported).

Behavioral notes:

- If the requested base item count cannot be satisfied under the provided per-prompt limits, the tool raises an error rather than silently clamping.

How sampling works:

- Determine per-request item count k by sampling uniformly from the integer range defined by `--random-mm-base-items-per-request` and `--random-mm-num-mm-items-range-ratio`, then clamp k to at most the sum of per-modality limits.
- For each of the k items, sample a bucket (H, W, T) according to the normalized probabilities in `--random-mm-bucket-config`, while tracking how many items of each modality have been added.
- If a modality (e.g., image) reaches its limit from `--random-mm-limit-mm-per-prompt`, all buckets of that modality are excluded and the remaining bucket probabilities are renormalized before continuing.
This should be seen as an edge case, and if this behavior can be avoided by setting `--random-mm-limit-mm-per-prompt` to a large number. Note that this might result in errors due to engine config `--limit-mm-per-prompt`.
- The resulting request contains synthetic image data in `multi_modal_data` (OpenAI Chat format). When `random-mm` is used with the OpenAI Chat backend, prompts remain text and MM content is attached via `multi_modal_data`.

</details>

### 🔬 マルチモーダルプロセッサのベンチマーク { #-multimodal-processor-benchmark }

Benchmark per-stage latency of the multimodal (MM) input processor pipeline, including the encoder forward pass. This is useful for profiling preprocessing bottlenecks in vision-language models.

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

The benchmark measures the following stages for each request:

| Stage | Description |
| ----- | ----------- |
| `get_mm_hashes_secs` | Time spent hashing multimodal inputs |
| `get_cache_missing_items_secs` | Time spent looking up the processor cache |
| `apply_hf_processor_secs` | Time spent in the HuggingFace processor |
| `merge_mm_kwargs_secs` | Time spent merging multimodal kwargs |
| `apply_prompt_updates_secs` | Time spent updating prompt tokens |
| `preprocessor_total_secs` | Total preprocessing time |
| `encoder_forward_secs` | Time spent in the encoder model forward pass |
| `num_encoder_calls` | Number of encoder invocations per request |

The benchmark also reports end-to-end latency (TTFT + decode time) per
request. Use `--metric-percentiles` to select which percentiles to report
(default: p99) and `--output-json` to save results.

#### 合成データを使った基本的な例（random-mm） { #basic-example-with-synthetic-data-random-mm }

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 50 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image": 3, "video": 0}' \
  --random-mm-bucket-config '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```

#### HuggingFace のデータセットを使う { #using-a-huggingface-dataset }

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 100
```

#### ウォームアップ、カスタムのパーセンタイル、JSON 出力 { #warmup-custom-percentiles-and-json-output }

```bash
vllm bench mm-processor \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --num-prompts 200 \
  --num-warmups 5 \
  --random-input-len 300 \
  --random-output-len 40 \
  --random-mm-base-items-per-request 1 \
  --metric-percentiles 50,90,95,99 \
  --output-json results.json
```

See [`vllm bench mm-processor`](../cli/bench/mm_processor.md) for the full argument reference.

</details>

### 埋め込みのベンチマーク { #embedding-benchmark }

Benchmark the performance of embedding requests in vLLM.

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

#### テキストの埋め込み { #text-embeddings }

生成モデルが Completions API や Chat Completions API を使うのとは異なり、Embeddings API を使うには
`--backend openai-embeddings` と `--endpoint /v1/embeddings` を設定します。

You can use any text dataset to benchmark the model, such as ShareGPT.

サーバーを起動します。

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

ベンチマークを実行します。

```bash
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model jinaai/jina-embeddings-v3 \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json
```

#### マルチモーダルの埋め込み { #multi-modal-embeddings }

Unlike generative models which use Completions API or Chat Completions API,
you should set `--endpoint /v1/embeddings` to use the Embeddings API. The backend to use depends on the model:

- CLIP: `--backend openai-embeddings-clip`
- VLM2Vec: `--backend openai-embeddings-vlm2vec`

その他のモデルについては、期待される指示の形式に合わせて [vllm/benchmarks/lib/endpoint_request_func.py](../../vllm/benchmarks/lib/endpoint_request_func.py) 内に独自の実装を追加してください。

モデルがサポートしていれば、任意のテキストまたはマルチモーダルのデータセットを使ってベンチマークできます。
たとえば ShareGPT と VisionArena を使って vision-language の埋め込みをベンチマークできます。

CLIP をサービングしてベンチマークします。

```bash
# Run this in another process
vllm serve openai/clip-vit-base-patch32

# Run these one by one after the server is up
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model openai/clip-vit-base-patch32 \
  --backend openai-embeddings-clip \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

VLM2Vec をサービングしてベンチマークします。

```bash
# Run this in another process
vllm serve TIGER-Lab/VLM2Vec-Full --runner pooling \
  --trust-remote-code \
  --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja

# Run these one by one after the server is up
# download dataset
# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
vllm bench serve \
  --model TIGER-Lab/VLM2Vec-Full \
  --backend openai-embeddings-vlm2vec \
  --endpoint /v1/embeddings \
  --dataset-name sharegpt \
  --dataset-path <your data path>/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model TIGER-Lab/VLM2Vec-Full \
  --backend openai-embeddings-vlm2vec \
  --endpoint /v1/embeddings \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat
```

</details>

### リランカーのベンチマーク { #reranker-benchmark }

vLLM におけるリランクリクエストの性能をベンチマークします。

<details class="admonition abstract" markdown="1">
<summary>Show more</summary>

生成モデルが Completions API や Chat Completions API を使うのとは異なり、Reranker API を使うには
`--backend vllm-rerank` と `--endpoint /v1/rerank` を設定します。

リランキングでサポートされるデータセットは `--dataset-name random-rerank` のみです。

サーバーを起動します。

```bash
vllm serve BAAI/bge-reranker-v2-m3
```

ベンチマークを実行します。

```bash
vllm bench serve \
  --model BAAI/bge-reranker-v2-m3 \
  --backend vllm-rerank \
  --endpoint /v1/rerank \
  --dataset-name random-rerank \
  --tokenizer BAAI/bge-reranker-v2-m3 \
  --random-input-len 512 \
  --num-prompts 10 \
  --random-batch-size 5
```

リランカーのモデルでは、`random_batch_size` 個の「ドキュメント」（各ドキュメントは
`random_input_len` に近いトークン数）を持つリクエストが `num_prompts / random_batch_size` 件作成されます。
上記の例では、各ドキュメントが 512 トークン近くを持つ「ドキュメント」を 5 個含むリランクリクエストが
2 件になります。

なお、`/v1/rerank` は埋め込みモデルでもサポートされています。埋め込みモデルで実行する場合は
`--no_reranker` も設定してください。この場合、クエリはサーバー側で個別のプロンプトとして扱われるため、
クエリという余分なプロンプトを考慮して `random_batch_size - 1` 個のドキュメントを送ります。
スループットの数値を正しく報告するためのトークンの計上も、あわせて調整されます。

</details>
