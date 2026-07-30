# 融合（fusion）の torch.compile パス { #fusion-torchcompile-passes }

vLLM は、最適化をモデル定義から切り離し、モデルコードの層の抽象を壊さないようにするため、コンパイル時に一連のカーネル / 演算子の融合を適用します（カスタムの [`torch.compile`](torch_compile.md) Inductor パス経由）。
これらの融合は [`PassConfig`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/compilation/#vllm.config.compilation.PassConfig) のフィールドで制御され、適切な[最適化レベル](optimization_levels.md)で自動的に有効になります。

## クイックリファレンス { #quick-reference }

下表は、各融合について制御するフラグ / 設定項目、融合される演算、既定で有効になるレベル、目安となる高速化率を示しています。
Fullgraph の列は、その融合がモデルのグラフ全体を参照する必要があるか（Inductor のパーティションまたは `splitting_ops=[]` のいずれかによる）を示し、最後の列は、その融合がすべての `num_tokens` で有効になるのか、小さい側 / 大きい側だけで有効になるのかを示します。

!!! info
    高速化率はモデル、バッチサイズ、ハードウェアに大きく依存します。
    手作業で性能を調整する場合は、必ず自分のユースケースで融合の有無を比較してベンチマークし、効果を確認してください。

| 融合                                                                         | `PassConfig` のフラグ            | 融合される演算                               | 既定で有効になるレベル                     | E2E の高速化        | Fullgraph | `num_tokens` |
| ------------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------- | ------------------------------ | ------------------ | --------- | ------------ |
| [AllReduce + RMSNorm](#allreduce--rmsnorm-fuse_allreduce_rms)                  | `fuse_allreduce_rms`         | All-reduce → RMSNorm (+residual_add) (→ quant) | O2 (Hopper/Blackwell + TP > 1) | 5-20%              | No        | Low          |
| [Attention + Quant](#attention--quantization-fuse_attn_quant)                  | `fuse_attn_quant`            | Attention output → FP8/NVFP4 quant             | Off by default                 | 3-7%               | Yes       | Always       |
| [MLA Attention + Quant](#attention--quantization-fuse_attn_quant)              | `fuse_attn_quant`            | MLA Attention output → FP8/NVFP4 quant         | Off by default                 | TBD                | Yes       | Always       |
| [RoPE + KV-Cache Update](#rope--kv-cache-update-fuse_rope_kvcache)             | `fuse_rope_kvcache`          | Rotary embedding → KV cache write              | O2 (ROCm/AITER only)           | 2-4%               | No        | Low          |
| [QK Norm + RoPE](#qk-norm--rope-enable_qk_norm_rope_fusion)                    | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → rotary embedding                 | Off by default                 | 2-3%               | No        | Low          |
| [Sequence Parallelism](#sequence-parallelism-enable_sp)                        | `enable_sp`                  | AllReduce → ReduceScatter + AllGather          | Off by default                 | Prereq for AsyncTP | Yes       | High         |
| [AsyncTP GEMM + collective](#asynctp-gemm--collective-overlap-fuse_gemm_comms) | `fuse_gemm_comms`            | GEMM → reduce-scatter / all-gather → GEMM      | Off by default                 | 7-10%              | Yes       | High         |
| [RMSNorm + Quant](#rmsnorm--quantization-fuse_norm_quant)                      | `fuse_norm_quant`            | RMSNorm (+residual add) → FP8/FP4 quant        | O1 (conditional)               | 1-4%               | No        | Always       |
| [SiLU+Mul + Quant](#silumul--quantization-fuse_act_quant)                      | `fuse_act_quant`             | SiLU+Mul activation → FP8/FP4 quant            | O1 (conditional)               | 1-4%               | No        | Always       |
| [RMSNorm + Padding](#rmsnorm--padding-fuse_act_padding)                        | `fuse_act_padding`           | Residual add + RMSNorm → padding               | O1 (ROCm/AITER only)           | TBD                | No        | Always       |
| [MLA Dual RMSNorm](#mla-dual-rmsnorm-fuse_mla_dual_rms_norm)                   | `fuse_mla_dual_rms_norm`     | Paired Q + KV RMSNorm (+ FP8 quant) → 1 kernel | O1 (ROCm/AITER only)           | 1-2%               | No        | Always       |

## サポートマトリクス { #support-matrix }

下表は、各プラットフォームで各融合がサポートする量子化方式を示しています。
**—** はそのプラットフォームでその融合が利用できないことを意味します。最新の状況と進行中の作業は追跡用 issue にあります:
[#36066](https://github.com/vllm-project/vllm/issues/36066)

| 融合                       | SM100（Blackwell）                        | SM90（Hopper）                            | SM89（Ada）                               | SM80（Ampere） | ROCm                                     |
| ---------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ------------- | ---------------------------------------- |
| `fuse_allreduce_rms`         | FP16/BF16, FP8 static, NVFP4             | FP16/BF16, FP8 static                    | —                                        | —             | —                                        |
| `fuse_attn_quant`\*          | FP8 static\*, NVFP4\*                    | FP8 static\*                             | FP8 static\*                             | —             | FP8 static\*                             |
| `fuse_attn_quant` (MLA)\*    | FP8 static\*, FP8 per-group\*, NVFP4\*   | FP8 static\*, FP8 per-group\*            | FP8 static\*, FP8 per-group\*            | —             | FP8 static\* (untested)                  |
| `fuse_rope_kvcache`          | —                                        | —                                        | —                                        | —             | FP16/BF16                                |
| `enable_qk_norm_rope_fusion` | FP16/BF16                                | FP16/BF16                                | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `enable_sp`                  | FP16/BF16, FP8 static†                   | FP16/BF16, FP8 static                    | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `fuse_gemm_comms`            | FP16/BF16, FP8 static†                   | FP16/BF16, FP8 static                    | FP16/BF16†                               | FP16/BF16†    | —                                        |
| `fuse_norm_quant`            | FP8 static, FP8 per-token, FP8 per-group | FP8 static, FP8 per-token, FP8 per-group | FP8 static, FP8 per-token, FP8 per-group | —             | FP8 static, FP8 per-token, FP8 per-group |
| `fuse_act_quant`             | FP8 static, NVFP4                        | FP8 static, FP8 per-group (128/64)       | FP8 static, FP8 per-group (128/64)       | —             | FP8 per-group                            |
| `fuse_act_padding`           | —                                        | —                                        | —                                        | —             | FP16/BF16                                |
| `fuse_mla_dual_rms_norm`     | —                                        | —                                        | —                                        | —             | BF16                                     |

\* `fuse_attn_quant` のサポートは、使用する attention バックエンドに依存します。すべてのバックエンドが融合された量子化出力をサポートするわけではありません。バックエンドごとの詳細は [`fuse_attn_quant` の節](#attention--quantization-fuse_attn_quant)を参照してください。

† `enable_sp` と `fuse_gemm_comms` は、現時点では SM90 についてのみ自動設定されます。
他のアーキテクチャでは `PassConfig.sp_min_token_num` を明示的に設定する必要があります。
SM100 でのサポートには、さらに `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` の設定も必要です。

## 融合の有効化 / 無効化 { #enabling-disabling-fusions }

融合は `CompilationConfig` に入れ子になった `PassConfig` を通じて公開されています。

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="...",
    optimization_level=2, # Default optimization level
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_allreduce_rms=False,  # disable a specific fusion
        )
    ),
)
```

任意の `vllm ...` コマンドで、コマンドラインのフラグにより融合を有効にすることもできます。

```bash
# Enable O2 defaults, but turn off allreduce fusion
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False

# The above is equivalent to the more verbose:
vllm serve meta-llama/Llama-3.1-8B-Instruct -O2 --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'

# Same syntax in other commands, e.g. vllm bench:
vllm bench latency --model=meta-llama/Llama-3.1-8B-Instruct -O2 -cc.pass_config.fuse_allreduce_rms=False
```

ユーザーが明示的に設定したフィールドは、常に最適化レベルの既定値より優先されます。

## 融合の詳細 { #fusion-details }

### AllReduce + RMSNorm（`fuse_allreduce_rms`） { #allreduce--rmsnorm-fuse_allreduce_rms }

!!! warning
    TP+DP と TP+PP の組み合わせは現在壊れています
    （[#34458](https://github.com/vllm-project/vllm/issues/34458) と
    [#35426](https://github.com/vllm-project/vllm/issues/35426)）。
    FlashInfer がインストールされた NVIDIA Hopper（SM90）と Blackwell（SM100）でのみサポートされます。

**融合される内容。** テンソル並列の all-reduce 集合通信を、それに続く残差加算、RMSNorm、そして任意で量子化のステップとともに、単一の FlashInfer / TRT-LLM の通信カーネルへ融合します。
この融合が有利なのは `num_tokens` が小さい場合のみであるため、コンパイル対象のうち小さい範囲でのみ適用されます。

対象となるパターン:

- `AllReduce → RMSNorm(+residual_add)`: FlashInfer を伴う CUDA sm90 以上
- `AllReduce → RMSNorm(+residual_add) → FP8 静的量子化`: FlashInfer を伴う CUDA sm90 以上
- `AllReduce → RMSNorm(+residual_add) → NVFP4 動的量子化`: FlashInfer を伴う CUDA sm100 以上

融合カーネルが使われるテンソルサイズの上限はハードウェアに依存し（SM90 / SM100 の TP=2 では 64 MB）、`PassConfig.fi_allreduce_fusion_max_size_mb` で設定できます。

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/allreduce_rms_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/allreduce_rms_fusion.py)
- FlashInfer の all-reduce: [`vllm/distributed/device_communicators/flashinfer_all_reduce.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/flashinfer_all_reduce.py)
- ベンチマーク: [`benchmarks/kernels/benchmark_fused_collective.py`](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_fused_collective.py)

### Attention + 量子化（`fuse_attn_quant`） { #attention--quantization-fuse_attn_quant }

!!! info
    `fuse_attn_quant` は現時点でどの最適化レベルでも既定では有効にならず、明示的に設定する必要があります。また、モデルのグラフ全体が参照可能である必要があります（Inductor のパーティションまたは `splitting_ops=[]`）。

**融合される内容。** attention の計算直後に出力の量子化を融合し、attention 出力の完全精度でのメモリ往復を省きます。この融合は、標準の `Attention` と（DeepSeek-V2/V3/R1 系のモデルが使う）`MLAAttention` の両方をサポートします。対象となるパターン:

`Attention → FP8 静的量子化`:

- `TRITON_ATTN`: CUDA、ROCm
- `FLASHINFER`: FlashInfer がインストールされた CUDA sm100 以上
- `ROCM_ATTN`: ROCm
- `ROCM_AITER_UNIFIED_ATTN`: AITER を伴う ROCm

`Attention → NVFP4 動的量子化`:

- `FLASHINFER`: FlashInfer がインストールされた CUDA sm100 以上

`MLAAttention → FP8 静的、FP8 グループ単位、NVFP4 動的量子化`

MLA の融合は `unified_mla_attention_with_output` op に対してグラフレベルで動作し、MLA のデコード / プレフィルのすべてのバックエンドの組み合わせで機能します。標準の `Attention` バックエンド（カーネルが FP8 出力を直接書き込む）とは異なり、現時点で MLA のプレフィル / デコードのバックエンドはいずれも FP8 / FP4 の直接出力をサポートしていません。融合は中間バッファへ書き込んだうえで別のステップで量子化するため、メモリ往復の削減はまだ実現していません。

!!! info
    MLA attention の融合は、現時点で測定可能な高速化をもたらすとは見込まれていません。
    MLA のプレフィル / デコードのカーネルが FP8 / FP4 の直接出力をサポートすれば改善します。

その他の attention バックエンドは、まだ融合された出力の量子化をサポートしていません。

**コードの場所。**

- パス（Attention）: [`vllm/compilation/passes/fusion/attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/attn_quant_fusion.py)
- パス（MLAAttention）: [`vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/mla_attn_quant_fusion.py)
- attention バックエンド: [`vllm/v1/attention/backends/`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/)

### RoPE + KV キャッシュ更新（`fuse_rope_kvcache`） { #rope--kv-cache-update-fuse_rope_kvcache }

!!! info
    ROCm / AITER 専用です。NVIDIA CUDA や CPU では利用できません。AITER の融合カーネルの性能上の問題により、この融合は既定で `num_tokens ≤ 256` の場合にのみ有効になります。
    この閾値は `PassConfig.rope_kvcache_fusion_max_token_num` で設定できます。

**融合される内容。** 回転位置埋め込み（RoPE）のカーネルと KV キャッシュへの scatter / 書き込みを単一のカーネルへ融合し、key と value のテンソルの読み書きが別々に発生するのを避けます。

必要な条件: AITER を有効にした AMD ROCm、`rotary_embedding` のカスタム op が有効であること（自動）、そして `kv_cache` の更新 op がグラフ内で参照可能であること（Inductor のグラフパーティションを使うか、`splitting_ops` から取り除く）。
これらの条件が満たされていれば、最適化レベル O1 以上でこの融合が自動的に有効になります。

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/rope_kvcache_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)

### シーケンス並列（`enable_sp`） { #sequence-parallelism-enable_sp }

**融合される内容。** all-reduce の集合通信を reduce-scatter + ローカルの RMSNorm + all-gather へ置き換え、シーケンス次元を TP ランク間で分割します。これによりグラフが再構成され、後続の AsyncTP のパスが reduce-scatter / all-gather を周囲の GEMM と融合できるようになります。

シーケンス並列そのものは性能を直接改善しません。AsyncTP のパス（`fuse_gemm_comms`）の前提条件です。SP は、デバイスの能力とモデルの `hidden_size` にもとづいて自動設定される最小トークン数の閾値を超える場合にのみ適用されます。現時点では、`hidden_size >= 8192` のモデルについて H100 / SM90 でのみ有効です。閾値は `PassConfig.sp_min_token_num` で設定できます。

一般的な変換は次のとおりです。

```text
Input → AllReduce → RMSNorm → Output
は次のようになります:
Input → ReduceScatter → ローカルの RMSNorm → AllGather → Output
```

対象となるパターン:

- 最初のブロック: `AllReduce → RMSNorm` → `ReduceScatter → RMSNorm → AllGather`
- 中間のブロック: `AllReduce → fused_add_RMSNorm` → `ReduceScatter → fused_add_RMSNorm → AllGather`
- いずれも任意で末尾に `→ FP8 静的量子化` が付く場合

必要な条件: `use_inductor_graph_partition=True`、**または** `tensor_parallel_size` で割り切れる静的サイズによる区分的コンパイル。

サポートされるハードウェア: NVIDIA CUDA でのみテスト済み（ROCm でも動作する可能性があります）。FP8 の all-gather には sm90 以上が必要です。

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### AsyncTP による GEMM + 集合通信のオーバーラップ（`fuse_gemm_comms`） { #asynctp-gemm--collective-overlap-fuse_gemm_comms }

!!! info
    `enable_sp=True` が必要です（自動的に有効になります）。シーケンス並列が適用されていない場合、このパスは何も行いません。

**融合される内容。** シーケンス並列がグラフを変換したあと、`torch.ops.symm_mem` の対称メモリのプリミティブを使い、GEMM カーネルを周囲の reduce-scatter（出力の射影）と all-gather（入力の射影）と融合し、通信と計算をオーバーラップさせます。
このオーバーラップが有利なのは `num_tokens` が大きい場合のみであるため、この融合（およびその前段の SP）は `PassConfig.sp_min_token_num` を超えるコンパイル対象の大きい範囲でのみ適用されます。

対象となるパターン:

- `GEMM → reduce-scatter` → `fused_matmul_reduce_scatter`
- `all-gather → GEMM` → `all_gather_matmul`
- 上記いずれのパターンの FP8 scaled 版

サポートされるハードウェア: 対称メモリ（`torch.distributed._symmetric_memory`）をサポートする NVIDIA CUDA。

B200 では fp8 の FlashInfer scaled MM に対するパターンマッチがサポートされないため、無効にする必要があります
（[#27893](https://github.com/vllm-project/vllm/issues/27893)）。

```shell
VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel ...
```

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/collective_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/collective_fusion.py)
- シーケンス並列のパス: [`vllm/compilation/passes/fusion/sequence_parallelism.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/sequence_parallelism.py)

### QK Norm + RoPE（`enable_qk_norm_rope_fusion`） { #qk-norm--rope-enable_qk_norm_rope_fusion }

!!! info
    回転位置埋め込みの前に Q と K へヘッド単位の RMSNorm を適用するモデル（Qwen など）にのみ適用されます。H100 での性能上の問題により、どの最適化レベルでも既定では有効になりません:
    [#34391](https://github.com/vllm-project/vllm/issues/34391)

**融合される内容。** QKV の分割 → reshape → Q/K の RMSNorm → reshape → 回転位置埋め込み、という一連の処理を単一の `fused_qk_norm_rope` CUDA カーネルへ融合します。

```text
# 融合前:
q, k, v = split(qkv)
q_norm = rms_norm(q.view(heads))
k_norm = rms_norm(k.view(kv_heads))
q_rope, k_rope = rotary_embedding(q_norm, k_norm, ...)

# 融合後:
fused_qk_norm_rope(qkv, ...)
```

サポートされるハードウェア: CUDA（sm80 以上）のみ。テスト済みなのは sm90 と sm100 のみです。

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/qk_norm_rope_fusion.py)
- CUDA カーネル: [`csrc/ops.h`](https://github.com/vllm-project/vllm/blob/main/csrc/ops.h)（`fused_qk_norm_rope`）

### RMSNorm + 量子化（`fuse_norm_quant`） { #rmsnorm--quantization-fuse_norm_quant }

!!! warning
    NVIDIA では、Inductor が生成する融合カーネルのほうが vLLM のカスタム CUDA カーネルより実際に高速です。
    そのため、この融合は `rms_norm` または `quant_fp8` のいずれかがカスタムカーネルを使っている場合にのみ有効になります。

**融合される内容。** カスタムの `rms_norm` / `fused_add_rms_norm` の演算と、それに続く量子化を単一の融合カーネルにまとめ、完全精度のアクティベーションテンソルの中間的な読み書きを省きます。
次の 2 つの形が融合されます。

- *素の RMSNorm + 量子化*: `rms_norm(x) → quant_fp8(y)`
- *fused-add RMSNorm + 量子化*: `fused_add_rms_norm(x, residual) → quant_fp8(y)` — 残差もインプレースで更新します。

なお、AITER の融合は現時点では `vllm.compilation.passes.fusion.rocm_aiter_fusion` の別パスにあります。

サポートされる量子化方式とハードウェアの組み合わせ:

- FP8 静的なテンソル単位: CUDA / HIP カーネル
- FP8 動的なトークン単位: CUDA / HIP カーネル、AITER
- FP8 動的なトークングループ単位（128/64）: CUDA / HIP カーネル、AITER

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/rms_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rms_quant_fusion.py)
- ROCm AITER のパス: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA / HIP カーネル: [`csrc/layernorm_quant_kernels.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu)

### SiLU+Mul + 量子化（`fuse_act_quant`） { #silumul--quantization-fuse_act_quant }

!!! warning
    `fuse_norm_quant` と同様に、NVIDIA では Inductor が生成する融合カーネルのほうがカスタム op より高速です。
    この融合は、`silu_and_mul` または `quant_fp8` のいずれかがカスタムカーネルを使っている場合、あるいは NVFP4 量子化のモデル（FP4 の量子化は常にカスタム op）でのみ有効になります。

**融合される内容。** gate-up 射影の活性化 `silu_and_mul` と、それに続く量子化を単一のカーネルへ融合し、活性化後の完全精度テンソルを実体化せずに済ませます。

なお、AITER の融合は `vllm.compilation.passes.fusion.rocm_aiter_fusion` の別パスにあります。

サポートされる量子化方式とハードウェアの組み合わせ:

- FP8 静的なテンソル単位: CUDA / HIP カーネル
- FP8 動的なグループ単位（128/64）: CUDA カーネル（sm89 以上。sm100 以上で DeepGemm を使う場合は有効になりません）
- NVFP4 動的: FlashInfer を伴う CUDA sm100 以上のみ
- FP8 トークングループ単位（128）: ROCm AITER のみ

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/act_quant_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/act_quant_fusion.py)
- ROCm AITER のパス: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)
- CUDA / HIP カーネル: [`csrc/quantization/`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/)
- SiLU+Mul+BlockQuant の融合カーネル: [`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu)

### RMSNorm + パディング（`fuse_act_padding`） { #rmsnorm--padding-fuse_act_padding }

!!! info
    ROCm / AITER 専用です。GPT-OSS 系のモデルを対象としています。

**融合される内容。** 残差加算 + RMSNorm と、それに続くパディング処理（下流の AITER Triton GEMM カーネルが要求する倍数まで hidden 次元をパディングする処理）を融合します。

必要な条件: AITER の RMSNorm を有効にした AMD ROCm。hidden size が 2880 で、AITER Triton GEMM が有効で*ない*場合に、最適化レベル O1 以上で既定で有効になります。

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)（`RocmAiterTritonAddRMSNormPadFusionPass`）

### MLA デュアル RMSNorm（`fuse_mla_dual_rms_norm`） { #mla-dual-rmsnorm-fuse_mla_dual_rms_norm }

!!! info
    ROCm / AITER 専用です。DeepSeek-V3 / Kimi-K2 の MLA attention を対象としています。

!!! note
    `rms_norm` のネイティブ実装が使われる場合（現時点では CUDA と ROCm の既定）、
    Inductor の組み込みの融合がこれらの norm のマージをすでに自動的に処理します。
    この明示的なパスは、Inductor 単体では融合できない AITER のカスタム `rms_norm` op が
    有効な場合を対象としています。

**融合される内容。** MLA attention における対になった `q_a_layernorm` と `kv_a_layernorm` の RMS norm 演算を、AITER 経由の単一の `fused_qk_rmsnorm` HIP カーネル呼び出しへ融合し、MLA 層あたりのカーネル起動を 2 回から 1 回へ減らします。

```text
# 融合前:
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_c  = rms_norm(q_c,  q_weight,  eps)
kv_c = rms_norm(kv_c, kv_weight, eps)

# 融合後:
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_normed, kv_normed = fused_mla_dual_rms_norm(
    q_c, q_weight, kv_c, kv_weight, eps1, eps2)
```

必要な条件: AITER を有効にした AMD ROCm。AITER が利用可能な場合、最適化レベル O1 以上で既定で有効になります。

**FP8 attention の場合（トークン単位の量子化）。** トークン単位 FP8 の `q_b_proj` を使う場合、FP8 量子化されるのは *q* の latent だけで、*kv* は bf16 のままです。
`RocmAiterRMSNormQuantFusionPass` はまず q 側を `rocm_aiter_rmsnorm_fused_dynamic_quant` に畳み込み、kv 側は素の `rms_norm` のまま残すため、上記の対称なパターンが崩れます。同じパスが続いてこの非対称な組をマッチさせ、`fused_mla_dual_rms_norm_per_token_quant` へ lowering します。

```text
# 融合前（q の norm+量子化は融合済み、kv はまだ素の rms_norm）:
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_fp8, q_scale = rocm_aiter_rmsnorm_fused_dynamic_quant(q_c, q_weight, eps, fp8)
kv_normed      = rms_norm(kv_c, kv_weight, eps)          # bf16

# 融合後:
q_c, kv_lora = split(projected, [q_dim, kv_dim])
kv_c, k_pe   = split(kv_lora,  [kv_c_dim, k_pe_dim])
q_fp8, q_scale, kv_normed = fused_mla_dual_rms_norm_per_token_quant(
    q_c, q_weight, kv_c, kv_weight, eps1, eps2)
```

**コードの場所。**

- パス: [`vllm/compilation/passes/fusion/rocm_aiter_fusion.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rocm_aiter_fusion.py)（`MLADualRMSNormFusionPass`、`MLADualRMSPerTokenQuantPattern`）
- カスタム op: [`vllm/_aiter_ops.py`](https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py)（`fused_mla_dual_rms_norm`、`fused_mla_dual_rms_norm_per_token_quant`）
- AITER カーネル: [`fused_qk_rmsnorm`](https://github.com/ROCm/aiter/pull/2442)、`fused_qk_rmsnorm_per_token_quant`

## 関連項目 { #see-also }

- [最適化レベル](optimization_levels.md) — 融合の既定値を設定する高水準のプリセット。
- [vLLM における torch.compile](torch_compile.md) — Inductor のパスのパイプラインの仕組み。
- [Attention バックエンド](attention_backends.md) — attention 固有のカーネル選択。
