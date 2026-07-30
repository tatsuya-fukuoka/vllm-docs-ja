# Fused MoE カーネルの機能 { #fused-moe-kernel-features }

このドキュメントの目的は、さまざまな MoE カーネル（モジュラー版・非モジュラー版の両方）を概観し、状況に応じて適切なカーネルの組み合わせを選びやすくすることです。モジュラーカーネルが使う all2all バックエンドの情報も含みます。

## Fused MoE モジュラー All2All バックエンド { #fused-moe-modular-all2all-backends }

`FusedMoE` 層のエキスパート並列（EP）を実装するために、複数の all2all 通信バックエンドが使われています。各 all2all バックエンドのインターフェースは、`FusedMoEPrepareAndFinalizeModular` のサブクラスとして提供されます。

次の表は、各バックエンドの主な特性（アクティベーションの形式、サポートする量子化方式、非同期対応）をまとめたものです。

出力アクティベーションの形式（standard または batched）は、`FusedMoEPrepareAndFinalizeModular` サブクラスの prepare ステップの出力に対応し、finalize ステップも同じ形式を必要とします。すべてのバックエンドの `prepare` メソッドは standard 形式のアクティベーションを受け取り、すべての `finalize` メソッドは standard 形式のアクティベーションを返します。形式の詳細は [Fused MoE モジュラーカーネル](./fused_moe_modular_kernel.md)のドキュメントを参照してください。

量子化の型と形式は、各 `FusedMoEPrepareAndFinalizeModular` クラスがどの量子化方式をサポートするかを示します。量子化は、all2all バックエンドがサポートする形式に応じて dispatch の前または後に行われます。たとえば deepep_high_throughput はブロック量子化された fp8 形式のみをサポートします。それ以外の形式では、より高い精度のまま dispatch し、その後で量子化することになります。各バックエンドの prepare ステップの出力は量子化後の型です。finalize ステップは通常、元のアクティベーションと同じ入力型を必要とします。たとえば元の入力が bfloat16 で量子化方式が per-tensor スケールの fp8 の場合、`prepare` は fp8 / per-tensor スケールのアクティベーションを返し、`finalize` は bfloat16 のアクティベーションを受け取ります。MoE 処理の各ステップにおけるアクティベーションの型と形式の詳細は、[Fused MoE モジュラーカーネル](./fused_moe_modular_kernel.md)の図を参照してください。量子化の型が指定されない場合、カーネルは float16 または bfloat16 で動作します。

非同期バックエンドは、DBO（Dual Batch Overlap）と共有エキスパートのオーバーラップ（combine ステップの実行中に共有エキスパートを計算する方式）の利用をサポートします。

一部のモデル（Llama など）では、topk==1 のときに topk の重みを出力アクティベーションではなく入力アクティベーションへ適用する必要があります。モジュラーカーネルでは、この機能は `FusedMoEPrepareAndFinalizeModular` サブクラスがサポートします。非モジュラーカーネルでは、このフラグの扱いは experts 関数側に委ねられます。

特に断りのない限り、バックエンドはコマンドライン引数 `--all2all-backend`（あるいは `ParallelConfig` の `all2all_backend` パラメータ）で制御します。`flashinfer` を除くすべてのバックエンドは EP+DP または EP+TP でのみ動作します。`Flashinfer` は EP なし、あるいは EP を伴わない DP でも動作します。

<style>
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}
</style>

| バックエンド | 出力アクティベーション形式 | 量子化の型 | 量子化の形式 | 非同期 | 入力に重みを適用 | サブクラス |
| ------- | ------------------ | ------------ | ------------- | ----- | --------------------- | --------- |
| naive | standard | all<sup>1</sup> | G,A,T | N | <sup>6</sup> | [`layer.py`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/layer/#vllm.model_executor.layers.fused_moe.layer.FusedMoE) |
| deepep_high_throughput | standard | fp8 | G(128),A,T<sup>2</sup> | Y | Y | [`DeepEPHTPrepareAndFinalize`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht/#vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht.DeepEPHTPrepareAndFinalize) |
| deepep_low_latency | batched | fp8 | G(128),A,T<sup>3</sup> | Y | Y | [`DeepEPLLPrepareAndFinalize`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll/#vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll.DeepEPLLPrepareAndFinalize) |
| flashinfer_nvlink_two_sided | standard | nvfp4,fp8 | G,A,T | N | N | [`FlashInferNVLinkTwoSidedPrepareAndFinalize`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided/#vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided.FlashInferNVLinkTwoSidedPrepareAndFinalize) |
| flashinfer_nvlink_one_sided | standard | nvfp4,bf16,mxfp8 | G,A,T | N | N | [`FlashInferNVLinkOneSidedPrepareAndFinalize`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided/#vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize) |

!!! info "表の凡例"
    1. すべての型: mxfp4、nvfp4、int4、int8、fp8
    2. A、T の量子化は dispatch のあとに行われます。
    3. 量子化はすべて dispatch のあとに行われます。
    4. `--moe-backend`（`flashinfer_cutlass` または `flashinfer_trtllm`）で制御します。
    5. これは何もしない dispatcher で、任意のモジュラー experts と組み合わせることで、dispatch も combine も行わないモジュラーカーネルを構成できます。環境変数では選択できません。主にテスト用途、あるいは experts のサブクラスを `fused_experts` API に適合させる目的で使います。
    6. experts 側の実装に依存します。

    ---

    - G - グループ単位
    - G(N) - ブロックサイズ N のグループ単位
    - A - アクティベーションのトークン単位
    - T - テンソル単位

モジュラーカーネルは、次の `FusedMoEMethodBase` クラスでサポートされています。

- [`ModelOptFp8MoEMethod`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/quantization/modelopt/#vllm.model_executor.layers.quantization.modelopt.ModelOptFp8MoEMethod)
- [`Fp8MoEMethod`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/quantization/fp8/#vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod)
- [`CompressedTensorsW4A4Nvfp4MoEMethod`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4/#vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4.CompressedTensorsW4A4Nvfp4MoEMethod)
- [`CompressedTensorsW8A8Fp8MoEMethod`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8/#vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8.CompressedTensorsW8A8Fp8MoEMethod)
- [`GptOssMxfp4MoEMethod`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/quantization/mxfp4/#vllm.model_executor.layers.quantization.mxfp4.GptOssMxfp4MoEMethod)
- [`UnquantizedFusedMoEMethod`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/#vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod)

## Fused Experts カーネル { #fused-experts-kernels }

量子化の型やアーキテクチャごとに、多数の MoE experts カーネル実装があります。そのほとんどは、ベースとなる Triton の [`fused_experts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/fused_moe/#vllm.model_executor.layers.fused_moe.fused_moe.fused_experts) 関数の一般的な API に従っています。多くはモジュラーカーネルのアダプタを備えており、互換性のある all2all バックエンドと組み合わせて使えます。次の表は、各 experts カーネルとその特性を一覧にしたものです。

各カーネルには、サポートされている入力アクティベーション形式のいずれかを渡す必要があります。カーネルによっては、異なるエントリポイントを通じて standard と batched の両形式をサポートします（`TritonExperts` と `BatchedTritonExperts` など）。batched 形式のカーネルが必要になるのは、現時点では特定の all2all バックエンド（`DeepEPLLPrepareAndFinalize` など）と組み合わせる場合だけです。

バックエンドのカーネルと同様に、各 experts カーネルがサポートする量子化形式も限られています。非モジュラーの experts では、アクティベーションは元の型のまま渡され、カーネル内部で量子化されます。モジュラーの experts では、アクティベーションがすでに量子化済みの形式であることが前提です。いずれの experts も、出力は元のアクティベーションの型で返します。

各 experts カーネルは 1 つ以上の活性化関数（silu、gelu など）をサポートし、中間結果に対して適用します。

バックエンドと同様に、一部の experts は topk の重みを入力アクティベーションへ適用することをサポートします。この表の該当列の内容は、非モジュラーの experts にのみ当てはまります。

ほとんどの experts には、`FusedMoEExpertsModular` のサブクラスとなる同等のモジュラーインターフェースが用意されています。

特定の `FusedMoEPrepareAndFinalizeModular` サブクラスと組み合わせて使うには、MoE カーネル側のアクティベーション形式・量子化の型・量子化の形式が互換である必要があります。

| カーネル | 入力アクティベーション形式 | 量子化の型 | 量子化の形式 | 活性化関数 | 入力に重みを適用 | モジュラー | ソース |
| ------ | ----------------- | ------------ | ------------- | ------------------- | --------------------- | ------- | ------ |
| triton | standard | all<sup>1</sup> | G,A,T | silu, gelu,</br>swigluoai,</br>silu_no_mul,</br>gelu_no_mul | Y | Y | [`fused_experts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/fused_moe/#vllm.model_executor.layers.fused_moe.fused_moe.fused_experts),</br>[`TritonExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/triton_moe/#vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts) |
| triton (batched) | batched | all<sup>1</sup> | G,A,T | silu, gelu | <sup>6</sup> | Y | [`BatchedTritonExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/fused_batched_moe/#vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.BatchedTritonExperts) |
| deep gemm | standard,</br>batched | fp8 | G(128),A,T | silu, gelu | <sup>6</sup> | Y | </br>[`DeepGemmExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe/#vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmExperts),</br>[`BatchedDeepGemmExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe/#vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe.BatchedDeepGemmExperts) |
| cutlass_fp4 | standard,</br>batched | nvfp4 | A,T | silu | Y | Y | [`CutlassExpertsFp4`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/cutlass_moe/#vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp4) |
| cutlass_fp8 | standard,</br>batched | fp8 | A,T | silu, gelu | Y | Y | [`CutlassExpertsFp8`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/cutlass_moe/#vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassExpertsFp8),</br>[`CutlasBatchedExpertsFp8`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/cutlass_moe/#vllm.model_executor.layers.fused_moe.experts.cutlass_moe.CutlassBatchedExpertsFp8) |
| flashinfer | standard | nvfp4,</br>fp8 | T | <sup>5</sup> | N | Y | [`FlashInferExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe/#vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe.FlashInferExperts) |
| gpt oss triton | standard | N/A | N/A | <sup>5</sup> | Y | Y | [`triton_kernel_fused_experts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe/#vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.triton_kernel_fused_experts),</br>[`OAITritonExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe/#vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe.OAITritonExperts) |
| marlin | standard,</br>batched | <sup>3</sup> / N/A | <sup>3</sup> / N/A | silu,</br>swigluoai | Y | Y | [`fused_marlin_moe`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/marlin_moe/#vllm.model_executor.layers.fused_moe.experts.marlin_moe.fused_marlin_moe),</br>[`MarlinExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/marlin_moe/#vllm.model_executor.layers.fused_moe.experts.marlin_moe.MarlinExperts),</br>[`BatchedMarlinExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/marlin_moe/#vllm.model_executor.layers.fused_moe.experts.marlin_moe.BatchedMarlinExperts) |
| trtllm | standard | mxfp4,</br>nvfp4 | G(16),G(32) | <sup>5</sup> | N | Y | [`TrtLlmMxfp4ExpertsMonolithic`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe/#vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsMonolithic),</br>[`TrtLlmMxfp4ExpertsModular`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe/#vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe.TrtLlmMxfp4ExpertsModular),</br>[`TrtLlmNvFp4ExpertsMonolithic`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe/#vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsMonolithic),</br>[`TrtLlmNvfp4ExpertsModular`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe/#vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe.TrtLlmNvFp4ExpertsModular) |
| hpc | standard | fp8 | G(128),T | silu | Y | Y | [`HPCExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/hpc_moe/#vllm.model_executor.layers.fused_moe.hpc_moe.HPCExperts) |
| rocm aiter moe | standard | mxfp4,</br>fp8 | G(32),G(128),A,T | silu, gelu,</br>swigluoai | Y | N | `rocm_aiter_fused_experts`,</br>`AiterExperts` |
| cpu_fused_moe | standard | N/A | N/A | silu | N | N | [`CPUFusedMOE`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/cpu_fused_moe/#vllm.model_executor.layers.fused_moe.cpu_fused_moe.CPUFusedMOE) |
| naive batched<sup>4</sup> | batched | int8,</br>fp8 | G,A,T | silu, gelu | <sup>6</sup> | Y | [`NaiveBatchedExperts`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/fused_moe/experts/fused_batched_moe/#vllm.model_executor.layers.fused_moe.experts.fused_batched_moe.NaiveBatchedExperts) |

!!! info "表の凡例"
    1. すべての型: mxfp4、nvfp4、int4、int8、fp8
    2. triton と deep gemm の experts をまとめる dispatcher のラッパーです。型・形状・量子化パラメータにもとづいて選択します。
    3. uint4、uint8、fp8、fp4
    4. batched 形式をサポートする素朴な experts 実装です。主にテストに使われます。
    5. `activation` パラメータは無視され、代わりに既定で SwiGlu が使われます。
    6. モジュラーカーネルと組み合わせて使う場合にのみ扱われる、またはサポートされます。

## モジュラーカーネルの「ファミリー」 { #modular-kernel-families }

次の表は、組み合わせて動作することを意図したモジュラーカーネルの「ファミリー」を示しています。動作する可能性はあるものの未検証の組み合わせもあります（flashinfer と他の fp8 experts の組み合わせなど）。

| バックエンド | `FusedMoEPrepareAndFinalizeModular` のサブクラス | `FusedMoEExpertsModular` のサブクラス |
| ------- | ---------------------------------------------- | ----------------------------------- |
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`,</br>`TritonExperts`,</br>`TritonOrDeepGemmExperts`,</br>`CutlassExpertsFp8`, </br>`MarlinExperts` |
| deepep_low_latency | `DeepEPLLPrepareAndFinalize` | `BatchedDeepGemmExperts`,</br>`BatchedTritonExperts`,</br>`CutlassBatchedExpertsFp8`,</br>`BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |
