# Attention バックエンドの機能サポート { #attention-backend-feature-support }

このドキュメントは `tools/pre_commit/generate_attention_backend_docs.py` によって自動生成されています。`AttentionBackend.validate_configuration()` のチェックにもとづき、登録された各 Attention バックエンドの機能サポート状況を示します。

**このファイルを手動で編集しないでください。** 再生成するには次のコマンドを実行します。

```bash
python tools/pre_commit/generate_attention_backend_docs.py
```

## Attention バックエンドの指定 { #setting-the-attention-backend }

### コマンドライン { #command-line }

コマンドラインからバックエンドを指定する方法は 2 つあります。

**方法 1: `--attention-backend` を使う（シンプル）**

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

**方法 2: `--attention-config.backend` / `-ac.backend` を使う（構造化された設定）**

```bash
# Dot notation
vllm serve <model> --attention-config.backend FLASH_ATTN
vllm serve <model> -ac.backend FLASH_ATTN

# JSON format
vllm serve <model> --attention-config '{"backend": "FLASH_ATTN"}'
vllm serve <model> -ac '{"backend": "FLASH_ATTN"}'
```

> **注:** `--attention-backend` と `--attention-config.backend` は排他的です。どちらか一方のみを使ってください。

### Python API { #python-api }

`LLM` クラスと一緒に `AttentionConfig` を使います。

```python
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Method 1: Using AttentionConfig with enum
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
)

# Method 2: Using attention_backend parameter with string
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    attention_backend="FLASH_ATTN",
)
```

## バックエンド選択の挙動 { #backend-selection-behavior }

### 手動での選択 { #manual-selection }

`--attention-backend` または `AttentionConfig` でバックエンドを明示的に指定した場合の挙動です。

1. そのバックエンドが、使用する構成（モデルの dtype、ヘッドサイズ、compute capability など）に対して**検証**されます。
2. バックエンドがその構成を**サポートしていない**場合、具体的な理由とともにエラーが送出されます。
3. 妥当であれば、そのバックエンドが使われます。

互換性のないバックエンドを選択した場合のエラー例は次のとおりです。

```text
ValueError: Selected backend FLASHMLA is not valid for this configuration.
Reason: ['compute capability not supported']
```

### 自動選択 { #automatic-selection }

バックエンドを指定しない場合（既定）の挙動です。

1. vLLM は**優先度順**にバックエンドを走査します（後述の表を参照）。
2. 各バックエンドが、使用する構成に対して検証されます。
3. **最初に互換性が確認されたバックエンド**が選択されます。
4. 互換性のあるバックエンドがない場合、すべてのバックエンドと非互換の理由を列挙したエラーが送出されます。

## バックエンドの優先度（CUDA） { #backend-priority-cuda }

バックエンドが明示的に選択されていない場合、vLLM はこれらの優先度順のリストから、最初に互換性のあるバックエンドを選びます。

優先度は **1 が最高**（最初に試される）です。

### 標準的な Attention（MHA、MQA、GQA） { #standard-attention-mha-mqa-gqa }

**Blackwell（SM 10.x）:**

| 優先度 | バックエンド |
| -------- | ------- |
| 1 | `FLASHINFER` |
| 2 | `FLASH_ATTN` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
| 5 | `TURBOQUANT` |

**Ampere / Hopper（SM 8.x-9.x）:**

| 優先度 | バックエンド |
| -------- | ------- |
| 1 | `FLASH_ATTN` |
| 2 | `FLASHINFER` |
| 3 | `TRITON_ATTN` |
| 4 | `FLEX_ATTENTION` |
| 5 | `TURBOQUANT` |

### MLA Attention（DeepSeek 方式） { #mla-attention-deepseek-style }

**Blackwell（SM 10.x）:**

| 優先度 | バックエンド |
| -------- | ------- |
| 1 | `FLASHINFER_MLA` |
| 2 | `TOKENSPEED_MLA` |
| 3 | `CUTLASS_MLA` |
| 4 | `FLASH_ATTN_MLA` |
| 5 | `FLASHMLA` |
| 6 | `TRITON_MLA` |
| 7 | `FLASHINFER_MLA_SPARSE`**\*** |
| 8 | `FLASHMLA_SPARSE` |

> **\*** スパース MLA では、FP8 の KV キャッシュの場合は常に `FLASHINFER_MLA_SPARSE` が優先されます。BF16 の KV キャッシュでは、クエリヘッド数が少ない場合（16 以下）は `FLASHINFER_MLA_SPARSE` が、それ以外では `FLASHMLA_SPARSE` が優先されます。
>
> **注:** ROCm と CPU のプラットフォームは独自の選択ロジックを持ちます。詳細は各プラットフォームのドキュメントを参照してください。

## 凡例 { #legend }

| 列 | 説明 |
| ------ | ----------- |
| **Dtypes** | サポートされるモデルのデータ型（fp16、bf16、fp32） |
| **KV Dtypes** | サポートされる KV キャッシュのデータ型（`auto`、`fp8`、`fp8_e4m3` など） |
| **Block Sizes** | サポートされる KV キャッシュのブロックサイズ（%N は N の倍数を意味します） |
| **Head Sizes** | サポートされる Attention のヘッドサイズ |
| **Sink** | Attention sink のサポート（StreamingLLM 向け） |
| **Non-Causal** | デコーダモデルにおける非因果（双方向）Attention のサポート |
| **Sparse** | スパース Attention のサポート（MLA のみ） |
| **MM Prefix** | マルチモーダルのプレフィックスに対する full attention のサポート |
| **DCP** | デコードのコンテキスト並列（`--decode-context-parallel-size`）のサポート |
| **Attention Types** | サポートされる Attention のパターン（Decoder、Encoder、Enc-Dec） |
| **Compute Cap.** | 必要な CUDA の compute capability（CUDA 以外のバックエンドでは N/A） |

**記号:** ✅ = サポート、❌ = 非サポート

## 標準的な Attention（MHA、MQA、GQA）のバックエンド { #standard-attention-mha-mqa-gqa-backends }

| Backend | Version | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | --------- | --- | --------------- | ------------ |
| `CPU_ATTN` | | fp16, bf16, fp32 | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | 32, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512 | ❌ | ✅ | ❌ | ❌ | All | N/A |
| `FLASHINFER` | Native† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | ❌ | ✅ | ❌ | ✅ | Decoder | 8.x-9.x |
| `FLASHINFER` | XQA† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | ❌ | ❌ | ❌ | ✅ | Decoder | 9.0 |
| `FLASHINFER` | trtllm-gen† | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `nvfp4` | 16, 32, 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | ✅ | ✅ | ❌ | ✅ | Decoder | 10.x |
| `FLASH_ATTN` | FA2* | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ✅ | ❌ | ✅ | All | ≥8.0 |
| `FLASH_ATTN` | FA3* | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | Any | ✅ | ✅ | ❌ | ✅ | All | 9.x |
| `FLASH_ATTN` | FA4* | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ✅ | ✅ | ❌ | ✅ | All | ≥10.0 |
| `FLASH_ATTN_DIFFKV` | | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ✅ | Decoder | Any |
| `FLEX_ATTENTION` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ✅ | ✅ | ❌ | Decoder, Encoder Only | Any |
| `HPC_ATTN` | | fp16, bf16 | `auto`, `bfloat16`, `fp8_e4m3` | 64 | 128 | ❌ | ❌ | ❌ | ❌ | Decoder | ≥9.0 |
| `ROCM_AITER_FA` | | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 16, 32 | 64, 128, 256 | ✅ | ✅ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_UNIFIED_ATTN` | | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | Any | ✅ | ❌ | ✅ | ❌ | All | N/A |
| `ROCM_ATTN` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %16 | 32, 64, 80, 96, 128, 160, 192, 224, 256 | ❌ | ✅ | ✅ | ❌ | Decoder, Encoder, Encoder Only | N/A |
| `TRITON_ATTN` | | fp16, bf16, fp32 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `int4_per_token_head`, `int8_per_token_head`, `fp8_per_token_head` | %16 | Any | ✅ | ✅ | ✅ | ❌ | All | Any |
| `TRITON_ATTN_DIFFKV` | | fp16, bf16 | `auto`, `bfloat16` | Any | Any | ❌ | ❌ | ❌ | ❌ | Decoder | Any |
| `TURBOQUANT` | | fp16, bf16 | `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc` | 16, 32, 64, 128 | Any | ❌ | ❌ | ❌ | ❌ | Decoder | Any |

> **†** FlashInfer Native は通常の FlashInfer の経路です。XQA は FlashInfer の TRTLLM デコード API を通じて公開される SM90 のデコード経路です。trtllm-gen は SM100 で使われ、sink をサポートします。XQA / trtllm-gen は `--attention-config.use_trtllm_attention=0` で無効にできます。
>
> **\*** FlashAttention のバージョンは `--attention-config.flash_attn_version=2`、`3`、`4` で指定します。既定は SM100 以上（Blackwell）では FA4、SM90（Hopper）では FA3、それ以外では FA2 です。

## MiniMax M3 のスパース Attention バックエンド { #minimax-m3-sparse-attention-backends }

MiniMax M3 のスパース（「lightning indexer」）層で使われるブロックスパースの GQA バックエンドです。モデルから直接組み込まれており、上記の自動選択の優先度リストには含まれません。lightning indexer が KV ブロックにスコアを付け、上位 k 個のブロック（および固定の初期 / ローカルブロック）が選択され、Attention はそれらのブロックのみを参照します。インデックス用のキーは別のサイドキャッシュに保持されます。

| Backend | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | --------- | --- | --------------- | ------------ |
| `MINIMAX_M3_SPARSE` | bf16, fp16 | `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | 128 | 128 | ❌ | ❌ | ❌ | ❌ | Decoder | Any |

## MLA（Multi-head Latent Attention）のバックエンド { #mla-multi-head-latent-attention-backends }

MLA では、プレフィルとデコードのフェーズで別々のバックエンドを使います。

### プレフィル用バックエンド { #prefill-backends }

プレフィル用のバックエンドを明示的に選択するには、`-ac.mla_prefill_backend=<BACKEND>`（`FLASH_ATTN`、`FLASHINFER` など）を使います。指定しない場合、プレフィル用バックエンドはハードウェアと構成にもとづいて実行時に自動選択されます。

| バックエンド | 説明 | Dtypes | Compute Cap. | 備考 |
| ------- | ----------- | ------ | ------------ | ----- |
| `FLASH_ATTN`‡ | FlashAttention varlen (FA2/FA3/FA4) | fp16, bf16 | Any | (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) (FA2/FA3/FA4) or (qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128) (FA2/FA3/FA4) or (qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256) (FA2/FA3 only) |
| `TRTLLM_RAGGED` | TensorRT-LLM ragged attention | fp16, bf16 | 10.x | (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) or (qk_nope_head_dim=192, qk_rope_head_dim=64, v_head_dim=256) only |
| `FLASHINFER` | FlashInfer CUTLASS backend | fp16, bf16 | 10.x | (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) only |
| `TOKENSPEED_MLA` | | fp16, bf16 | 10.x | (qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128) only |

> **‡** 自動選択ではまず FlashAttention が試されます。Blackwell（SM100）でのフォールバック順は、TRT-LLM Ragged、FlashInfer、TokenSpeed MLA です。それ以外の GPU では FlashAttention のみが検討されます。

### デコード用バックエンド { #decode-backends }

MLA のデコード用バックエンドは、標準の `-ac.backend=<BACKEND>` 引数（`FLASHMLA`、`TRITON_MLA` など）で選択します。

| Backend | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | Sparse | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | ------ | --------- | --- | --------------- | ------------ |
| `CUTLASS_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 128 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `FLASHINFER_MLA_SPARSE_SM120` | bf16 | `auto`, `fp8`, `fp8_e4m3`, `fp8_ds_mla` | 64, 256 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | 12.x |
| `FLASHMLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 9.x-10.x |
| `FLASHMLA_SPARSE` | bf16 | `auto`, `bfloat16`, `fp8_ds_mla` | 64 | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x-10.x |
| `FLASH_ATTN_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16` | %16 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 9.x |
| `FLASH_ATTN_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16` | 64 | Any | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x |
| `ROCM_AITER_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | %1 | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | 1, 64 | Any | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | N/A |
| `ROCM_AITER_TRITON_MLA` | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
| `TOKENSPEED_MLA` | fp16, bf16 | `fp8`, `fp8_e4m3` | 32, 64 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | 10.x |
| `TRITON_MLA` | fp16, bf16 | `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` | %16 | Any | ❌ | ❌ | ❌ | ❌ | ✅ | Decoder | Any |
| `XPU_MLA_SPARSE` | fp16, bf16 | `auto`, `float16`, `bfloat16` | Any | 576 | ❌ | ❌ | ✅ | ❌ | ❌ | Decoder | Any |

### DeepSeek V4 のデコード用バックエンド { #deepseek-v4-decode-backends }

DeepSeek V4 のスパース MLA は独自のデコード用バックエンドを使い、`--attention-backend=<BACKEND>`（`FLASHMLA_SPARSE_DSV4`、`FLASHINFER_MLA_SPARSE_DSV4` など）で選択します。これらは V4 のスパースインデックスのパイプライン（compressor + SWA + indexer、256 トークンのブロック、ヘッド 512）を共有します。NVIDIA での既定は、SM12x では `FLASHINFER_MLA_SPARSE_DSV4`、それ以外のサポート対象の CUDA アーキテクチャでは `FLASHMLA_SPARSE_DSV4` です。

| Backend | Dtypes | KV Dtypes | Block Sizes | Head Sizes | Sink | Non-Causal | Sparse | MM Prefix | DCP | Attention Types | Compute Cap. |
| ------- | ------ | --------- | ----------- | ---------- | ---- | ---------- | ------ | --------- | --- | --------------- | ------------ |
| `FLASHINFER_MLA_SPARSE_DSV4` | bf16 | `auto`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_ds_mla` | 256 | 512 | ✅ | ❌ | ✅ | ❌ | ❌ | Decoder | 10.x, 12.x |
| `FLASHMLA_SPARSE_DSV4` | bf16 | `auto`, `fp8_ds_mla`, `fp8` | 256 | 512 | ✅ | ❌ | ✅ | ❌ | ❌ | Decoder | 9.x-10.x |
| `ROCM_FLASHMLA_SPARSE_DSV4` | fp16, bf16 | `auto` | Any | Any | ❌ | ❌ | ❌ | ❌ | ❌ | Decoder | N/A |
