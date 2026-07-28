# 動的な投機的デコーディング { #dynamic-speculative-decoding }

## 動的 SD が必要な理由 { #why-is-dynamic-sd-needed }

投機的デコーディング (SD) では、Decode 中に各系列について K 個のトークンを検証する必要があります。バッチサイズ (BS) が大きくなると実効的な BS は BS\*K となり、検証時の計算量が増えます。この BS\*K がある臨界点を超えると、SD はかえって Decode 速度 (TPOT) を悪化させます。動的 SD (DSD) は K を最適な値に調整することで、SD の利点を維持できるようにします。

## ユースケース { #use-cases }

* 同じデプロイで同時実行数が変動するワークロード。同時実行数が増えると K は小さくなります。
* RL のロールアウト。最初は BS が大きいものの、大量のトークンを生成する少数のロングテールのリクエストが残ってロールアウトの進行を妨げ、最終的に BS が小さくなる場合です。この場合、ロールアウトの終盤で K は大きくなります。

## `--speculative-config` のスキーマ { #speculative-config-schema }

動的 SD を使うには、SD 手法の設定にリストのリストである `num_speculative_tokens_per_batch_size` を追加します。各要素は `[start_bs, end_bs, optimal_K]` で、同時実行数が `[start_bs, end_bs]` の範囲にあるとき `optimal_K` 個のドラフトトークンを使うことを意味します。例:

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

この設定は次を意味します。

* 同時実行数が [1, 64] の範囲では K=3 を使う
* 同時実行数が [65, 128] の範囲では K=1 を使う
* 同時実行数が [129, 512] の範囲では K=0 を使う（ドラフトトークンを生成しない）

## オンラインの例 { #online-examples }

### 動的 SD + Eagle ドラフタ { #dynamic-sd-eagle-drafter }

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### 動的 SD + Eagle3 ドラフタ { #dynamic-sd-eagle3-drafter }

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## 制限事項 { #limitations }

* Eagle、Eagle-3、DFlash で検証済みです。他の SD 手法はそのままでは動作しない可能性があります
* Full CUDA グラフは Model Runner V2 でのみ動作します。MRv1 ではこの機能と組み合わせると piecewise の CUDA グラフのみ対応します
* データ並列（`--data-parallel-size > 1`）とは併用できません。各 DP ランクが独立にスケジューリングするため、ランクごとに異なる K が選ばれ、DP の集団通信が食い違ってデッドロックする可能性があります。DP が有効な場合、vLLM は自動的に `num_speculative_tokens_per_batch_size` を無効化し、静的な `num_speculative_tokens` の値にフォールバックします。
