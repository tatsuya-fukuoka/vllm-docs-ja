# オンライン量子化 { #online-quantization }

オンライン量子化を使うと、BF16 / FP16 のモデルを読み込む時点で、Linear 層と MoE 層の重みをより低い精度（FP8 など）に量子化できます。量子化済みのチェックポイントやキャリブレーションデータは必要ありません。重みはモデルの読み込み時に変換され、活性値は各 forward パスで動的にスケーリングされます。

## クイックスタート { #quick-start }

`quantization` パラメータにスキーム名を渡します。

```python
from vllm import LLM

# Per-tensor FP8 quantization (one scale per weight tensor)
llm = LLM("meta-llama/Llama-3.1-8B", quantization="fp8_per_tensor")

# Per-block FP8 quantization (128x128 block scaling for weights and 1x128 block scaling for activations)
llm = LLM("meta-llama/Llama-3.1-8B", quantization="fp8_per_block")

# MXFP8 quantization for weights and activations
llm = LLM("meta-llama/Llama-3.1-8B", quantization="mxfp8")
```

CLI では次のようにします。

```bash
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_tensor
vllm serve meta-llama/Llama-3.1-8B --quantization fp8_per_block
vllm serve meta-llama/Llama-3.1-8B --quantization mxfp8
```

## サポートされるスキーム { #supported-schemes }

| スキーム | 重みのレシピ | 活性値のレシピ | 備考 |
| ------ | ------------- | ------------------ | ----- |
| `fp8_per_tensor` | fp8_e4m3 のデータ、fp32 のテンソル単位スケール | fp8_e4m3 のデータ、fp32 のテンソル単位スケール | 一部の GPU（Ada、Hopper）では、性能向上のため Linear 層の活性値にトークン単位のスケーリングを使います |
| `fp8_per_block` | fp8_e4m3 のデータ、fp32 の 128x128 ブロック単位スケール | fp8_e4m3 のデータ、fp32 の 1x128 ブロック単位スケール | |
| `mxfp8` | fp8_e4m3 のデータ、e8m0 の 1x32 ブロック単位スケール | fp8_e4m3 のデータ、e8m0 の 1x32 ブロック単位スケール | w8a8 には SM 100 以上（Blackwell 以降）が必要です。それ以外の GPU では w8a16 にフォールバックします |

## 高度な設定 { #advanced-configuration }

より細かく制御したい場合は、`quantization_config` の辞書を使います。

### スキーマ { #schema }

```yaml
quantization_config:
  linear:
    weight: <name>      # see QUANT_KEY_NAMES in vllm/config/quantization.py
    activation: <name>
  moe:
    weight: <name>
    activation: <name>
  ignore: [<layer-name-or-regex>, ...]
```

`linear` と `moe` には、`{weight, activation}` の完全な辞書か、単なる文字列を指定できます。文字列はまず `--quantization` の短縮名として解決され（対応する層種別のスロットが使われます）、次に `QUANT_KEY_NAMES` の重み形式名として解決されます。設定されていないフィールドは `--quantization` の短縮名の既定値にフォールバックし、量子化済みチェックポイントの場合はチェックポイントが宣言している内容にフォールバックします。

XPU では、ブロック単位でない FP8 の scaled-mm Linear 層は既定で W8A16 になります。`--linear-backend xpu` を指定すると W8A8 が強制されます。重みのみの量子化（W8A16）を明示的に選ぶには `--linear-backend xpu_woq` を使います。

CLI では、JSON と同じ形式、またはドット区切りのキーで指定できます。

```bash
vllm serve <model> --quantization-config '{"moe":{"activation":"mxfp8"}}'
vllm serve <model> --quantization-config.moe.activation mxfp8
```

### 量子化済みチェックポイントに対する活性値のオーバーライド { #activation-overrides-on-already-quantized-checkpoints }

チェックポイントの時点で量子化されているモデルでは、`quantization_config` を使うと、埋め込まれた重みとは独立に活性値の形式を選べます。サポートされるオーバーライドはチェックポイントごとに異なります。現時点では MXFP4 の MoE チェックポイント（gpt-oss）に対応しており、FP8 の活性値を選択できます。

```bash
vllm serve openai/gpt-oss-20b --quantization-config.moe.activation mxfp8
```

特定のカーネル系統に固定したい場合は `--moe-backend` と組み合わせてください。

### dense 層と MoE 層で別々のスキームを使う { #separate-schemes-for-dense-and-moe-layers }

`linear` と `moe` のフィールドを使うと、dense な Linear 層と MoE のエキスパート層に別々の量子化スキームを適用できます。それぞれ、完全な指定辞書か、オンライン量子化の短縮名（例: `"fp8_per_block"`）または重み形式名（例: `"fp8_per_block_static"`）を表す文字列を受け付けます。設定されていないフィールドは短縮名の既定値にフォールバックします。

```python
from vllm import LLM

# Linear: per-block FP8; MoE: per-tensor FP8 (inherited from the shorthand)
llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "linear": "fp8_per_block",
    },
)
```

あるいは次のようにもできます。

```python
from vllm import LLM

# Linear: per-tensor FP8 (inherited); MoE: per-block FP8
llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "moe": "fp8_per_block",
    },
)
```

### 特定の層を量子化から除外する { #excluding-layers-from-quantization }

特定の層をスキップするには `ignore` パラメータを使います。層名の完全一致と、`re:` を前置した正規表現パターンを受け付けます。

```python
from vllm import LLM

llm = LLM(
    "ibm-granite/granite-3.0-1b-a400m-base",
    quantization="fp8_per_tensor",
    quantization_config={
        "ignore": [
            # exact layer name
            "model.layers.1.self_attn.o_proj",
            # regex: skip all QKV projections
            "re:.*[qkv]_proj",
        ],
    },
)
```

!!! note
    融合された層（`q_proj`、`k_proj`、`v_proj` を融合した `qkv_proj` など）では、ignore の
    パターンは融合後の名前ではなく、**融合前** のシャード名（`q_proj`、`k_proj`、`v_proj`）に
    一致させる必要があります。
