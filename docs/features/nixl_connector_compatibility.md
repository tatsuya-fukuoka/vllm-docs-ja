# NixlConnector 互換性マトリクス { #nixlconnector-compatibility-matrix }

このページは、**NixlConnector を使ったプレフィル分離**における機能の互換性をまとめたものです。一般的な使い方は [NixlConnector 利用ガイド](nixl_connector_usage.md)を、プレフィル分離の概要は[プレフィル分離](disagg_prefill.md)を参照してください。

!!! note
    このページは現時点のコードベースの状態を反映しており、機能の進展に伴って変わる可能性があります。
    🟠 や ❌ が付いた項目は、追跡用 issue にリンクしている場合があります。今後の機能開発については
    [NIXL コネクタのロードマップ](https://github.com/vllm-project/vllm/issues/33702)を参照してください。

**凡例:**

- ✅ = 完全にサポート
- 🟠 = 部分的にサポート（脚注を参照）
- ❌ = 非対応
- ❔ = 不明 / 未検証
- 🚧 = 作業中

!!! info "すべてのモデルで共通してサポートされる機能"
    NixlConnector による PD 分離サービングでは、次の機能が**すべての**モデルアーキテクチャで動作します。

    [Chunked Prefill](../configuration/optimization.md#chunked-prefill) |
    [APC (Prefix Caching)](automatic_prefix_caching.md) |
    [Data Parallel](../serving/data_parallel_deployment.md) |
    CUDA graph |
    Logprobs |
    Prompt Logprobs |
    [Prompt Embeds](prompt_embeds.md) |
    Multiple NIXL backends (UCX, GDS, LIBFABRIC, etc.)

## モデルアーキテクチャ × 機能 { #model-architecture-x-capability }

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| モデル種別 | <abbr title="Basic Prefill/Decode disaggregation">Basic PD</abbr> | <abbr title="Speculative Decoding">Spec Decode</abbr> | <abbr title="Heterogeneous Tensor Parallelism (P TP != D TP)">Hetero TP</abbr> | <abbr title="Cross-layer blocks optimization">Cross-layer blocks</abbr> | <abbr title="Sliding Window Attention">SWA</abbr> | <abbr title="CPU host buffer offload (e.g. TPU)">Host buffer</abbr> | <abbr title="Different block sizes on P and D">Hetero block size</abbr> |
| - | - | - | - | - | - | - | - |
| Dense Transformers | ✅ | ✅<sup>1</sup> | ✅ | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| MLA (e.g. DeepSeek-V2/V3) | ✅ | ✅<sup>1</sup> | 🟠<sup>4</sup> | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| Sparse MLA (e.g. DeepSeek-V3.2) | ✅ | ✅<sup>1</sup> | 🟠<sup>4</sup> | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| Hybrid SSM / Mamba | ✅ | ❔ | 🚧<sup>5</sup> | ❌ | ✅ | ✅ | ❌<sup>6</sup> |
| MoE | ✅ | ✅<sup>1</sup> | ✅ | ✅<sup>2</sup> | ✅ | ✅ | 🟠<sup>3</sup> |
| Multimodal | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ |
| Encoder-Decoder | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

<sup>1</sup> P と D のインスタンスは同じ投機（speculation）設定を使う必要があります。

<sup>2</sup> `FLASH_ATTN` または `FLASHINFER` バックエンドと、`HND` の KV キャッシュレイアウトの**両方**が必要です。`--kv-transfer-config '{"kv_connector_extra_config": {"enable_cross_layers_blocks": "True"}}'` で有効にします。

<sup>3</sup> HMA が**不要**な場合（つまり hybrid でないモデル）にのみサポートされます。ブロック ID は自動的に再マッピングされます。サポートされるのは P のブロックサイズ < D のブロックサイズの場合のみです。

<sup>4</sup> MLA の KV キャッシュは TP ワーカー間で複製されるため、異種 TP は動作しますがヘッドの分割は行われません。P TP > D TP の場合、読み出しは 1 回だけ実行されます（冗長なランクはスキップされます）。D TP > P TP も動作します。

<sup>5</sup> Hybrid SSM（Mamba）系のモデルは**同種の TP**（`P TP == D TP`）を必要とします。Mamba 層では異種 TP はまだサポートされていません。

<sup>6</sup> HMA（hybrid モデルで必要）は、リモート側と異なるブロックサイズをサポートしていません。

## 設定に関する注意 { #configuration-notes }

### P と D で一致させる必要があるもの { #what-must-match-between-p-and-d }

既定では、ハンドシェイク時に**互換性ハッシュ**が検証されます。P と D のインスタンスは次の点で一致している必要があります。

- vLLM のバージョンと NIXL コネクタのバージョン
- モデル（アーキテクチャ、dtype、KV ヘッド数、ヘッドサイズ、隠れ層の数）
- Attention バックエンド
- KV キャッシュの dtype（`cache_dtype`）

!!! warning
    `--kv-transfer-config '{"kv_connector_extra_config": {"enforce_handshake_compat": false}}'` で
    ハッシュ検証を無効にできますが、自己責任で行ってください。

### P と D で異なっていても問題ないもの { #what-can-safely-differ-between-p-and-d }

- `tensor-parallel-size`（異種 TP。上記のモデルごとの制約に従います）
- `block-size`（異種ブロックサイズ。上記の制約に従います）
- KV キャッシュのブロック数（各インスタンスの利用可能メモリによって決まります）

### KV キャッシュのレイアウト { #kv-cache-layout }

- NixlConnector は、転送性能を最適化するため既定で **`HND`** レイアウトを使います（MLA 以外のモデル）。
- `NHD` レイアウトもサポートされますが、異種 TP でのヘッド分割は**できません**。
- 実験的な `HND` ↔ `NHD` の permute: `--kv-transfer-config '{"enable_permute_local_kv": true}'` で有効にします。HMA との併用はサポートされていません。

### 量子化された KV キャッシュ { #quantized-kv-cache }

[量子化 KV キャッシュ](quantization/quantized_kvcache.md)（FP8 など）を使う場合、P と D の両インスタンスが**同じ** `cache_dtype` を使う必要があります。cache dtype が食い違うと、ハンドシェイク時の互換性ハッシュ検証に失敗します。

- **静的量子化**（スケールをチェックポイントから読み込む）: ✅ サポート。各インスタンスがモデルのチェックポイントから独立にスケールを読み込みます。
- **動的量子化**（スケールを実行時に計算する）: ❌ 非対応。ブロック単位のスケールは KV キャッシュデータと一緒には転送されません。
- **packed レイアウトのスケール**（スケールを重みとインラインで保持）: ✅ サポート。スケールは KV キャッシュのブロックと一緒に転送されます。
