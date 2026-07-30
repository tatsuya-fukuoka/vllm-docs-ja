# 量子化 KV キャッシュ { #quantized-kv-cache }

## FP8 KV キャッシュの概要 { #fp8-kv-cache-overview }

大規模言語モデルを扱ううえで、メモリの効率的な利用は非常に重要です。KV（Key-Value）キャッシュを FP8 形式に量子化すると、そのメモリ使用量を大幅に削減できます。この最適化により、より多くのトークンをメモリに保持できるようになり、スループットの向上とより長いコンテキストウィンドウのサポートにつながります。

> **注:** FP8 KV キャッシュと Flash Attention 3 バックエンドを併用する場合、Attention の演算も量子化された（FP8）領域で行われます。この構成では、key と value に加えて query も FP8 に量子化されます。

### サポートされる FP8 KV キャッシュの量子化方式 { #supported-fp8-kv-cache-quantization-schemes }

vLLM は FP8 KV キャッシュについて、主に 2 つの量子化戦略をサポートしています。

- **テンソル単位の量子化:**  
  Q、K、V の各テンソルにそれぞれ 1 つのスケールを適用します（`q/k/v_scale = [1]`）。
- **Attention ヘッド単位の量子化:**  
  各スケールが 1 つの Attention ヘッドに対応します（`q_scale = [num_heads]`、`k/v_scale = [num_kv_heads]`）。

> **注:**  
> Attention ヘッド単位の量子化は現時点で **Flash Attention バックエンドでのみ**利用でき、**llm-compressor** が提供するキャリブレーション経路が必要です。

### スケールのキャリブレーション方法 { #scale-calibration-approaches }

vLLM では、量子化スケールの計算方法を次の 3 通りから選べます。

1. **キャリブレーションなし（既定のスケール）:**  
   すべての量子化スケールが `1.0` に設定されます。  
   _設定方法:_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=False
   ```

2. **ランダムトークンによるキャリブレーション（その場で実施）:**  
   ウォームアップ中にランダムなトークンのバッチ 1 つからスケールが自動的に推定され、その後固定されます。  
   _設定方法:_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=True
   ```

3. **［推奨］データセットによるキャリブレーション（llm-compressor 経由）:**  
   精度を最大化するため、厳選したキャリブレーション用データセットを使ってスケールを推定します。  
   これには [llm-compressor](https://github.com/vllm-project/llm-compressor) ライブラリが必要です。  
   _後述の例を参照してください。_

#### `kv_cache_dtype` のその他の選択肢 { #additional-kv_cache_dtype-options }

- `kv_cache_dtype="auto"`: モデルの既定のデータ型を使用します
- `kv_cache_dtype="fp8_e4m3"`: CUDA 11.8 以降と ROCm（AMD GPU）でサポート
- `kv_cache_dtype="fp8_e5m2"`: CUDA 11.8 以降でサポート

### 特定の層を KV キャッシュ量子化から除外する { #skipping-specific-layers-from-kv-cache-quantization }

一部の Attention 層の種類（sliding-window など）は、KV キャッシュの量子化に対してより敏感です。`--kv-cache-dtype-skip-layers` フラグを使うと、指定した層はモデル本来の dtype のまま残し、それ以外の層は選択した量子化 dtype で扱えます。このフラグは層のインデックス、または層の種類名を受け付けます。

```bash
# Skip every sliding-window attention layer.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window

# Skip specific layer indices.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 1 23
```

プログラムから使う場合は次のとおりです。

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["sliding_window"],
)
```

---

## 例 { #examples }

### 1. キャリブレーションなし（`kv_cache_dtype="fp8"`、`calculate_kv_scales=False`） { #1-no-calibration-kv_cache_dtypefp8-calculate_kv_scalesfalse }

すべての量子化スケールが 1.0 に設定されます。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. ランダムトークンによるキャリブレーション（`kv_cache_dtype="fp8"`、`calculate_kv_scales=True`） { #2-random-token-calibration-kv_cache_dtypefp8-calculate_kv_scalestrue }

ウォームアップ中にトークンのバッチ 1 つからスケールが自動的に推定されます。

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 3. **［推奨］データセットを使ったキャリブレーション（`llm-compressor` を利用）** { #3-recommended-calibration-using-a-dataset-with-llm-compressor }

最高品質の量子化を得るには、`llm-compressor` を使ってデータセットに対しキャリブレーションすることを推奨します。これにより、Attention ヘッド単位の量子化のような高度な方式も利用できます。

#### 必要なパッケージのインストール { #install-the-required-package }

```bash
pip install llmcompressor
```

#### 例: Llama の Attention と KV キャッシュを FP8 に量子化する { #example-quantize-llama-attention-kv-cache-to-fp8 }

```python
"""
Quantize Llama attention + KV cache to FP8 (choose either 'tensor' or 'attn_head' strategy)
using llm-compressor one-shot calibration.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # or "attn_head"
NUM_CALIB_SAMPLES = 512   # Good starting value
MAX_SEQ_LEN = 2048

# -----------------------------
# Helpers
# -----------------------------
def process_and_tokenize(example, tokenizer: AutoTokenizer):
    """Convert chat messages to tokens."""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],  # Quantize queries: q_scale
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,           # Quantize KV cache: k/v_scale
    )

# -----------------------------
# Main
# -----------------------------
def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
```

より詳しく最新の例については、[`llm-compressor` の公式サンプル](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache)を参照してください。
