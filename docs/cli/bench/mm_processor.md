# vllm bench mm-processor { #vllm-bench-mm-processor }

## 概要 { #overview }

`vllm bench mm-processor` は、視覚言語モデルのマルチモーダル入力プロセッサのパイプラインを
プロファイリングします。HuggingFace のプロセッサからエンコーダーの forward までを段階ごとに
計測するため、前処理のボトルネックを特定したり、画像の解像度や項目数の違いが
リクエスト全体の時間にどう影響するかを把握したりできます。

データソースは 2 種類サポートしています。合成のランダムなマルチモーダル入力 (`random-mm`) と
HuggingFace のデータセット (`hf`) です。計測前にウォームアップのリクエストを実行し、
結果が安定するようにしています。

## クイックスタート { #quick-start }

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

## 計測される段階 { #measured-stages }

| 段階 | 説明 |
| ----- | ----------- |
| `get_mm_hashes_secs` | マルチモーダル入力のハッシュ計算にかかった時間 |
| `get_cache_missing_items_secs` | プロセッサキャッシュの参照にかかった時間 |
| `apply_hf_processor_secs` | HuggingFace のプロセッサでの処理時間 |
| `merge_mm_kwargs_secs` | マルチモーダルの kwargs のマージにかかった時間 |
| `apply_prompt_updates_secs` | プロンプトトークンの更新にかかった時間 |
| `preprocessor_total_secs` | 前処理の合計時間 |
| `encoder_forward_secs` | エンコーダーモデルの forward にかかった時間 |
| `num_encoder_calls` | リクエストあたりのエンコーダー呼び出し回数 |

ベンチマークはリクエストごとのエンドツーエンドのレイテンシ（TTFT + Decode 時間）も出力します。
報告するパーセンタイルは `--metric-percentiles` で選択でき（既定は p99）、
`--output-json` で結果を保存できます。

HF データセット、ウォームアップ、JSON 出力などのより詳しい例は
[ベンチマーク CLI — マルチモーダルプロセッサのベンチマーク](../../benchmarking/cli.md#multimodal-processor-benchmark)を参照してください。

## JSON 形式の CLI 引数 { #json-cli-arguments }

--8<-- "docs/cli/json_tip.inc.md"

## 引数 { #arguments }

--8<-- "docs/generated/argparse/bench_mm_processor.inc.md"
