# 機能 { #features }

## 互換性マトリクス { #compatibility-matrix }

以下の表は、同時に使えない機能の組み合わせと、一部ハードウェアでの対応状況を示しています。

記号の意味は次のとおりです。

- ✅ = 完全に互換
- 🟠 = 部分的に互換
- ❌ = 非互換
- ❔ = 不明・未定

!!! note
    リンク付きの ❌ や 🟠 をたどると、その機能・ハードウェアの組み合わせに関する追跡 Issue を確認できます。

### 機能 × 機能 { #feature-x-feature }

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

| 機能 | [CP](../configuration/optimization.md#chunked-prefill) | [APC](automatic_prefix_caching.md) | [LoRA](lora.md) | [SD](speculative_decoding/README.md) | CUDA graph | [pooling](../models/pooling_models/README.md) | <abbr title="Encoder-Decoder Models">enc-dec</abbr> | <abbr title="Logprobs">logP</abbr> | <abbr title="Prompt Logprobs">prmpt logP</abbr> | <abbr title="Async Output Processing">async output</abbr> | multi-step | <abbr title="Multimodal Inputs">mm</abbr> | best-of | beam-search | [prompt-embeds](prompt_embeds.md) |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| [CP](../configuration/optimization.md#chunked-prefill) | ✅ | | | | | | | | | | | | | | |
| [APC](automatic_prefix_caching.md) | ✅ | ✅ | | | | | | | | | | | | | |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | | | | | | | | | | | | |
| [SD](speculative_decoding/README.md) | ✅ | ✅ | ❌ | ✅ | | | | | | | | | | | |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | |
| [pooling](../models/pooling_models/README.md) | 🟠\* | 🟠\* | ✅ | ❌ | ✅ | ✅ | | | | | | | | | |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr> | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ❌ | [❌](https://github.com/vllm-project/vllm/issues/7366) | ✅ | ✅ | ✅ | | | | | | | | |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | | | | | | |
| <abbr title="Async Output Processing">async output</abbr> | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | | | | | |
| multi-step | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | | | | |
| [mm](multimodal_inputs.md) | ✅ | ✅ | [🟠](https://github.com/vllm-project/vllm/pull/4194)<sup>^</sup> | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | | |
| best-of | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ✅ | ✅ | | |
| beam-search | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/6137) | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | [❌](https://github.com/vllm-project/vllm/issues/7968) | ❔ | ✅ | ✅ | |
| [prompt-embeds](prompt_embeds.md) | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❔ | ❔ | ✅ | ❔ | ❔ | ✅ |

\* チャンク化 Prefill とプレフィックスキャッシュは、causal attention を用いる last-token プーリングまたは all プーリングにのみ適用できます。  
<sup>^</sup> LoRA はマルチモーダルモデルの言語バックボーンにのみ適用できます。  

### 機能 × ハードウェア { #feature-x-hardware }

| 機能 | Volta | Turing | Ampere | Ada | Hopper | CPU | AMD | Intel GPU |
| ------- | ----- | ------ | ------ | --- | ------ | --- | --- | --------- |
| [CP](../configuration/optimization.md#chunked-prefill) | [❌](https://github.com/vllm-project/vllm/issues/2729) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [APC](automatic_prefix_caching.md) | [❌](https://github.com/vllm-project/vllm/issues/3687) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [LoRA](lora.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [SD](speculative_decoding/README.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| CUDA graph | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/26970) |
| [pooling](../models/pooling_models/README.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| <abbr title="Encoder-Decoder Models">enc-dec</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| [mm](multimodal_inputs.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [prompt-embeds](prompt_embeds.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ |
| <abbr title="Logprobs">logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| <abbr title="Prompt Logprobs">prmpt logP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| <abbr title="Async Output Processing">async output</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| multi-step | ✅ | ✅ | ✅ | ✅ | ✅ | [❌](https://github.com/vllm-project/vllm/issues/8477) | ✅ | ✅ |
| best-of | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| beam-search | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

!!! note
    Google TPU での機能の対応状況については、[TPU-Inference Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)（英語）を参照してください。
