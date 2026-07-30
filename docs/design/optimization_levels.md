# 最適化レベル { #optimization-levels }

## 概要 { #overview }

vLLM には 4 段階の最適化レベル（`-O0`、`-O1`、`-O2`、`-O3`）があり、起動時間と性能のトレードオフを選べます。

- `-O0`: 最適化なし。起動が最速ですが、性能は最も低くなります。
- `-O1`: 高速な最適化。シンプルなコンパイルと高速な融合（fusion）、および PIECEWISE の cudagraph を有効にします。
- `-O2`: 既定の最適化。コンパイル範囲の追加、融合の追加、FULL_AND_PIECEWISE の cudagraph を有効にします。
- `-O3`: 積極的な最適化。現時点では `-O2` と同じですが、将来的により時間のかかる最適化や実験的な最適化が追加される可能性があります。

各最適化レベルの既定値は、対応するフラグを手動で設定することでも再現できます。ユーザーが明示的に設定したフラグは、最適化レベルの既定値より優先されます。

## 各レベルの概要と使用例 { #level-summaries-and-usage-examples }

```bash
# CLI usage
vllm serve RedHatAI/Llama-3.2-1B-FP8 -O1

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=2 # equivalent to -O2
)
```

### `-O0`: 最適化なし { #-o0-no-optimization }

可能な限り高速に起動します。オートチューニングもコンパイルも cudagraph も行いません。開発の初期段階やデバッグに適したレベルです。

設定内容:

- `-cc.cudagraph_mode=NONE`
- `-cc.mode=NONE`（結果として `-cc.custom_ops=["none"]` にもなります）
- `-cc.pass_config.fuse_...=False`（すべての融合が無効）
- `--kernel-config.enable_flashinfer_autotune=False`

### `-O1`: 高速な最適化 { #-o1-fast-optimization }

起動の速さを優先しつつ、コンパイルや cudagraph といった基本的な最適化は有効にします。起動を速くしたいけれども、自分のコードが cudagraph やコンパイルを壊していないことは確認しておきたい、という多くの開発シナリオでバランスの良いレベルです。

設定内容:

- `-cc.cudagraph_mode=PIECEWISE`
- `-cc.mode=VLLM_COMPILE`
- `--kernel-config.enable_flashinfer_autotune=True`

融合:

- `-cc.pass_config.fuse_norm_quant=True`*
- `-cc.pass_config.fuse_act_quant=True`*
- `-cc.pass_config.fuse_act_padding=True`†
- `-cc.pass_config.fuse_mla_dual_rms_norm=True`†

\* これらの融合は、いずれかの演算がカスタムカーネルを使っている場合のみ有効になります。そうでない場合は Inductor の融合のほうが優れています。</br>
† これらの融合は ROCm 専用で、AITER が必要です。

### `-O2`: 完全な最適化（既定） { #-o2-full-optimization-default }

起動時間が伸びることと引き換えに性能を優先します。本番ワークロードにはこのレベルを推奨しており、そのため既定値になっています。このレベルの融合は、コンパイル範囲が追加されるぶん時間が長くなる場合が_あります_。

設定内容（`-O1` に加えて）:

- `-cc.cudagraph_mode=FULL_AND_PIECEWISE`
- `-cc.pass_config.fuse_allreduce_rms=True`
- `-cc.pass_config.fuse_rope_kvcache=True`†

† これらの融合は ROCm 専用で、AITER が必要です。

### `-O3`: 積極的な最適化 { #-o3-aggressive-optimization }

現時点では `-O2` と同じですが、将来的により時間のかかる最適化や実験的な最適化が追加される可能性があります。

## トラブルシューティング { #troubleshooting }

### よくある問題 { #common-issues }

1. **起動時間が長すぎる**: `-O0` または `-O1` を使って起動を速くしてください
2. **コンパイルエラー**: `debug_dump_path` を使って追加のデバッグ情報を取得してください
3. **性能が出ない**: 本番では `-O2` を使っているか確認してください
