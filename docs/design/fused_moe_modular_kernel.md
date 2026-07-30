# Fused MoE モジュラーカーネル { #fused-moe-modular-kernel }

## はじめに { #introduction }

FusedMoEModularKernel の実装は[こちら](../../vllm/model_executor/layers/fused_moe/modular_kernel.py)にあります。

入力アクティベーションの形式にもとづき、FusedMoE の実装は大きく 2 種類に分類されます。

* Contiguous / Standard / Non-Batched
* Batched

!!! note
    このドキュメントでは、Contiguous、Standard、Non-Batched という用語を同じ意味で使います。

入力アクティベーションの形式は、使用する All2All Dispatch によって完全に決まります。

* Contiguous の場合、All2All Dispatch はアクティベーションを形状 (M, K) の連続したテンソルとして返し、あわせて形状 (M, num_topk) の TopK ID と TopK 重みを返します。例としては `DeepEPHTPrepareAndFinalize` を参照してください。
* Batched の場合、All2All Dispatch はアクティベーションを形状 (num_experts, max_tokens, K) のテンソルとして返します。ここでは、同じエキスパートに割り当てられたアクティベーション / トークンがひとまとめにバッチ化されます。テンソルのすべての要素が有効なわけではない点に注意してください。アクティベーションのテンソルには通常、サイズ `num_experts` の `expert_num_tokens` テンソルが付随し、`expert_num_tokens[i]` が i 番目のエキスパートに割り当てられた有効なトークン数を示します。例としては `DeepEPLLPrepareAndFinalize` を参照してください。

FusedMoE の処理は、Contiguous と Batched のどちらの場合も、一般に下図のように複数の演算から構成されます。

![FusedMoE Non-Batched](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/fused_moe_modular_kernel/fused_moe_non_batched.png)

![FusedMoE Batched](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/fused_moe_modular_kernel/fused_moe_batched.png)

!!! note
    演算という観点で見た Batched と Non-Batched の主な違いは、Permute / Unpermute 演算の有無です。それ以外の演算は共通です。

## 動機 { #motivation }

図から分かるように演算の数は多く、それぞれの演算にさまざまな実装があり得ます。有効な FusedMoE 実装を構成するための演算の組み合わせ方は、すぐに手に負えない規模になります。モジュラーカーネルのフレームワークは、演算を論理的なコンポーネントにまとめることでこの問題に対処します。この大まかな分類により、組み合わせを扱える規模に抑え、コードの重複も防げます。また、All2All Dispatch / Combine の実装を FusedMoE の実装から切り離し、それぞれ独立に開発・テストできるようにします。さらに、モジュラーカーネルのフレームワークは各コンポーネントに対応する抽象クラスを導入し、今後の実装のための明確な骨組みを提供します。

このドキュメントの以降では Contiguous / Non-Batched のケースに焦点を当てます。Batched のケースへの拡張は容易なはずです。

## モジュラーカーネルのコンポーネント { #modularkernel-components }

FusedMoEModularKernel は FusedMoE の処理を 3 つの部分に分割します。

1. TopKWeightAndReduce
2. FusedMoEPrepareAndFinalizeModular
3. FusedMoEExpertsModular

### TopKWeightAndReduce { #topkweightandreduce }

TopK 重みの適用と reduction のコンポーネントは、Unpermute 演算の直後、All2All Combine の直前で実行されます。Unpermute を担うのは `FusedMoEExpertsModular`、All2All Combine を担うのは `FusedMoEPrepareAndFinalizeModular` である点に注意してください。TopK 重みの適用と reduction を `FusedMoEExpertsModular` の中で行うことには利点がありますが、実装によっては `FusedMoEPrepareAndFinalizeModular` の側で行うことを選びます。この柔軟性を実現するために、TopKWeightAndReduce 抽象クラスがあります。

TopKWeightAndReduce の実装は[こちら](../../vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py)にあります。

`FusedMoEPrepareAndFinalizeModular::finalize()` メソッドは `TopKWeightAndReduce` 引数を受け取り、メソッド内部でそれを呼び出します。
`FusedMoEModularKernel` は `FusedMoEExpertsModular` と `FusedMoEPrepareAndFinalize` の実装のあいだの橋渡しとして働き、TopK 重みの適用と reduction をどちらで行うかを決定します。

* `FusedMoEExpertsModular` の実装が重みの適用と reduction を自前で行う場合、`FusedMoEExpertsModular::finalize_weight_and_reduce_impl` メソッドは `TopKWeightAndReduceNoOp` を返します。
* `FusedMoEExpertsModular` の実装が `FusedMoEPrepareAndFinalizeModular::finalize()` に重みの適用と reduction を任せる場合、`FusedMoEExpertsModular::finalize_weight_and_reduce_impl` メソッドは `TopKWeightAndReduceContiguous` / `TopKWeightAndReduceNaiveBatched` / `TopKWeightAndReduceDelegate` を返します。

### FusedMoEPrepareAndFinalizeModular { #fusedmoeprepareandfinalizemodular }

`FusedMoEPrepareAndFinalizeModular` 抽象クラスは `prepare`、`prepare_no_receive`、`finalize` の各関数を公開します。
`prepare` 関数は、入力アクティベーションの量子化と All2All Dispatch を担当します。`prepare_no_receive` は、実装されている場合、`prepare` と同様ですが、他のワーカーからの結果受信を待ちません。代わりに、ワーカーの最終結果を待つために呼び出す必要がある「receiver」コールバックを返します。このメソッドはすべての `FusedMoEPrepareAndFinalizeModular` クラスでサポートされている必要はありませんが、利用できる場合は、最初の all to all 通信と他の処理をインターリーブする（共有エキスパートと fused experts をインターリーブするなど）ために使えます。`finalize` 関数は All2All Combine の呼び出しを担当します。さらに、`finalize` 関数は TopK 重みの適用と reduction を行う場合も行わない場合もあります（TopKWeightAndReduce の節を参照してください）。

![FusedMoEPrepareAndFinalizeModular のブロック](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/fused_moe_modular_kernel/prepare_and_finalize_blocks.png)

### FusedMoEExpertsModular { #fusedmoeexpertsmodular }

`FusedMoEExpertsModular` クラスは MoE 処理の中核が実行される場所です。`FusedMoEExpertsModular` 抽象クラスは、いくつかの重要な関数を公開します。

* apply()
* workspace_shapes()
* finalize_weight_and_reduce_impl()

#### apply()

`apply` メソッドでは、実装が次の処理を行います。

* Permute
* 重み W1 との matmul
* Act + Mul
* 量子化
* 重み W2 との matmul
* Unpermute
* 場合によっては TopK 重みの適用 + reduction

#### workspace_shapes()

FusedMoE の中核となる実装は一連の演算を行います。これらの演算ごとに個別に出力用メモリを確保するのは非効率です。そこで各実装には、workspace_shapes() メソッドの出力として、2 つのワークスペースの形状、ワークスペースのデータ型、そして FusedMoE の出力形状を宣言することが求められます。この情報は `FusedMoEModularKernel::forward()` でワークスペースのテンソルと出力テンソルを確保するために使われ、`FusedMoEExpertsModular::apply()` メソッドへ渡されます。ワークスペースは、FusedMoE 実装内で中間バッファとして利用できます。

#### finalize_weight_and_reduce_impl()

TopK 重みの適用と reduction を `FusedMoEExpertsModular::apply()` の内部で行ったほうが効率的な場合もあります。例は[こちら](https://github.com/vllm-project/vllm/pull/20228)を参照してください。こうした実装を可能にするために `TopKWeightAndReduce` 抽象クラスが用意されています。TopKWeightAndReduce の節を参照してください。
`FusedMoEExpertsModular::finalize_weight_and_reduce_impl()` は、その実装が `FusedMoEPrepareAndFinalizeModular::finalize()` に使ってほしい `TopKWeightAndReduce` オブジェクトを返します。

![FusedMoEExpertsModular のブロック](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/fused_moe_modular_kernel/fused_experts_blocks.png)

### FusedMoEModularKernel { #fusedmoemodularkernel }

`FusedMoEModularKernel` は `FusedMoEPrepareAndFinalizeModular` と `FusedMoEExpertsModular` のオブジェクトから構成されます。
`FusedMoEModularKernel` の疑似コード / スケッチは次のとおりです。

```py
class FusedMoEModularKernel:
    def __init__(self,
                 prepare_finalize: FusedMoEPrepareAndFinalizeModular,
                 fused_experts: FusedMoEExpertsModular):

        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def forward(self, DP_A):

        Aq, A_scale, _, _, _ = self.prepare_finalize.prepare(DP_A, ...)

        workspace13_shape, workspace2_shape, _, _ = self.fused_experts.workspace_shapes(...)

        # allocate workspaces
        workspace_13 = torch.empty(workspace13_shape, ...)
        workspace_2 = torch.empty(workspace2_shape, ...)

        # execute fused_experts
        fe_out = self.fused_experts.apply(Aq, A_scale, workspace13, workspace2, ...)

        # war_impl is an object of type TopKWeightAndReduceNoOp if the fused_experts implementations
        # performs the TopK Weight Application and Reduction.
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()

        output = self.prepare_finalize.finalize(fe_out, war_impl,...)

        return output
```

## ハウツー { #how-to }

### FusedMoEPrepareAndFinalizeModular 型を追加する方法 { #how-to-add-a-fusedmoeprepareandfinalizemodular-type }

一般に、FusedMoEPrepareAndFinalizeModular 型は All2All Dispatch / Combine の実装（カーネル）に支えられています。たとえば次のとおりです。

* DeepEPHTPrepareAndFinalize 型は DeepEP の High-Throughput All2All カーネルに支えられています。
* DeepEPLLPrepareAndFinalize 型は DeepEP の Low-Latency All2All カーネルに支えられています。

#### ステップ 1: All2All マネージャを追加する { #step-1-add-an-all2all-manager }

All2All マネージャの役割は、All2All のカーネル実装をセットアップすることです。`FusedMoEPrepareAndFinalizeModular` の実装は通常、Dispatch / Combine 関数を呼び出すために、All2All マネージャからカーネル実装の「ハンドル」を取得します。All2All マネージャの実装は[こちら](../../vllm/distributed/device_communicators/all2all.py)を参照してください。

#### ステップ 2: FusedMoEPrepareAndFinalizeModular 型を追加する { #step-2-add-a-fusedmoeprepareandfinalizemodular-type }

この節では、`FusedMoEPrepareAndFinalizeModular` 抽象クラスが公開する各関数の意味を説明します。

`FusedMoEPrepareAndFinalizeModular::prepare()`: prepare メソッドは量子化と All2All Dispatch を実装します。通常、対応する All2All マネージャの Dispatch 関数が呼び出されます。

`FusedMoEPrepareAndFinalizeModular::has_prepare_no_receive()`: このサブクラスが `prepare_no_receive` を実装しているかどうかを示します。既定値は False です。

`FusedMoEPrepareAndFinalizeModular::prepare_no_receive()`: prepare_no_receive メソッドは量子化と All2All Dispatch を実装します。dispatch 処理の結果を待たず、代わりに最終結果を待つために呼び出せる thunk を返します。通常、対応する All2All マネージャの Dispatch 関数が呼び出されます。

`FusedMoEPrepareAndFinalizeModular::finalize()`: 場合によって TopK 重みの適用と reduction を行い、All2All Combine を実行します。通常、対応する All2AllManager の Combine 関数が呼び出されます。

`FusedMoEPrepareAndFinalizeModular::activation_format()`: prepare メソッド（つまり All2All dispatch）の出力が Batched の場合は `FusedMoEActivationFormat.BatchedExperts` を、そうでない場合は `FusedMoEActivationFormat.Standard` を返します。

`FusedMoEPrepareAndFinalizeModular::topk_indices_dtype()`: TopK ID のデータ型です。一部の All2All カーネルは TopK ID のデータ型に厳密な要件を持ちます。この要件は `FusedMoe::select_experts` 関数へ伝えられ、そこで尊重されます。厳密な要件がない場合は None を返します。

`FusedMoEPrepareAndFinalizeModular::max_num_tokens_per_rank()`: 一度に All2All Dispatch へ渡されるトークン数の上限です。

`FusedMoEPrepareAndFinalizeModular::num_dispatchers()`: dispatch 単位の総数です。この値によって Dispatch の出力サイズが決まります。Dispatch の出力は形状 (num_local_experts, max_num_tokens, K) を持ちます。ここで max_num_tokens = num_dispatchers() * max_num_tokens_per_rank() です。

自分の All2All 実装に近い既存の `FusedMoEPrepareAndFinalizeModular` 実装を選び、参考にすることをおすすめします。

### FusedMoEExpertsModular 型を追加する方法 { #how-to-add-a-fusedmoeexpertsmodular-type }

FusedMoEExpertsModular は FusedMoE 処理の中核を担います。抽象クラスが公開する各関数とその意味は次のとおりです。

`FusedMoEExpertsModular::activation_formats()`: サポートする入力・出力のアクティベーション形式（Contiguous / Batched）を返します。

`FusedMoEExpertsModular::supports_expert_map()`: 実装が expert map をサポートする場合に True を返します。

`FusedMoEExpertsModular::workspace_shapes()` /
`FusedMoEExpertsModular::finalize_weight_and_reduce_impl` /
`FusedMoEExpertsModular::apply`: 上記の `FusedMoEExpertsModular` の節を参照してください。

### FusedMoEModularKernel の初期化 { #fusedmoemodularkernel-initialization }

`FusedMoEMethodBase` クラスには、`FusedMoEModularKernel` オブジェクトの生成を分担する 3 つのメソッドがあります。

* maybe_make_prepare_finalize
* select_gemm_impl
* init_prepare_finalize

#### maybe_make_prepare_finalize

`maybe_make_prepare_finalize` メソッドは、現在の all2all バックエンドにもとづいて適切な場合（EP + DP が有効な場合など）に `FusedMoEPrepareAndFinalizeModular` のインスタンスを構築する役割を担います。基底クラスのメソッドは、現時点では EP+DP のケース向けにすべての `FusedMoEPrepareAndFinalizeModular` オブジェクトを構築します。派生クラスはこのメソッドをオーバーライドして、別のシナリオ向けの prepare / finalize オブジェクトを構築できます。たとえば `ModelOptNvFp4FusedMoE` は EP+TP のケース向けに `FlashInferCutlassMoEPrepareAndFinalize` を構築できます。
実装は次を参照してください。

* `ModelOptNvFp4FusedMoE`

#### select_gemm_impl

`select_gemm_impl` メソッドは基底クラスでは未定義です。妥当かつ適切な `FusedMoEExpertsModular` オブジェクトを構築するメソッドを実装するのは、派生クラスの責任です。
次の派生クラスの実装を参照してください。

* `UnquantizedFusedMoEMethod`
* `CompressedTensorsW8A8Fp8MoEMethod`
* `CompressedTensorsW8A8Fp8MoECutlassMethod`
* `Fp8MoEMethod`
* `ModelOptNvFp4FusedMoE`


#### init_prepare_finalize

`init_prepare_finalize` メソッドは、入力と環境設定にもとづいて適切な `FusedMoEPrepareAndFinalizeModular` オブジェクトを作成します。続いて `select_gemm_impl` に問い合わせて適切な `FusedMoEExpertsModular` オブジェクトを取得し、`FusedMoEModularKernel` オブジェクトを構築します。

[init_prepare_finalize](https://github.com/vllm-project/vllm/blob/1cbf951ba272c230823b947631065b826409fa62/vllm/model_executor/layers/fused_moe/layer.py#L188) を参照してください。
**重要**: `FusedMoEMethodBase` の派生クラスは、`apply` メソッド内で `FusedMoEMethodBase::fused_experts` オブジェクトを使います。設定によって妥当な `FusedMoEModularKernel` オブジェクトを構築できる場合、その値で `FusedMoEMethodBase::fused_experts` を上書きします。これにより、派生クラスは実際にどの FusedMoE 実装が使われるかを意識せずに済みます。

### ユニットテストの書き方 { #how-to-unit-test }

`FusedMoEModularKernel` のユニットテストは [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) にあります。

このユニットテストは `FusedMoEPrepareAndFinalizeModular` と `FusedMoEPremuteExpertsUnpermute` の型のすべての組み合わせを走査し、互換性がある場合に正しさのテストを実行します。
`FusedMoEPrepareAndFinalizeModular` / `FusedMoEExpertsModular` の実装を追加する場合は、次のようにします。

1. [mk_objects.py](../../tests/kernels/moe/modular_kernel_tools/mk_objects.py) の `MK_ALL_PREPARE_FINALIZE_TYPES` と `MK_FUSED_EXPERT_TYPES` に、それぞれ実装の型を追加します。
2. [/tests/kernels/moe/modular_kernel_tools/common.py](../../tests/kernels/moe/modular_kernel_tools/common.py) の `Config::is_batched_prepare_finalize()`、`Config::is_batched_fused_experts()`、`Config::is_standard_fused_experts()`、
`Config::is_fe_16bit_supported()`、`Config::is_fe_fp8_supported()`、`Config::is_fe_block_fp8_supported()`
の各メソッドを更新します。

これにより、新しい実装がテストスイートに追加されます。

### `FusedMoEPrepareAndFinalizeModular` と `FusedMoEExpertsModular` の互換性を確認する方法 { #how-to-check-fusedmoeprepareandfinalizemodular-fusedmoeexpertsmodular-compatibility }

ユニットテストのファイル [test_modular_kernel_combinations.py](../../tests/kernels/moe/test_modular_kernel_combinations.py) は、単体のスクリプトとしても実行できます。
例: `python3 -m tests.kernels.moe.test_modular_kernel_combinations --pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts`
副次的な効果として、このスクリプトは `FusedMoEPrepareAndFinalizeModular` と `FusedMoEExpertsModular` の互換性の確認にも使えます。互換性のない型を指定して実行すると、スクリプトはエラーになります。

### プロファイルの取り方 { #how-to-profile }

[profile_modular_kernel.py](../../tests/kernels/moe/modular_kernel_tools/profile_modular_kernel.py) を参照してください。
このスクリプトは、互換性のある `FusedMoEPrepareAndFinalizeModular` と `FusedMoEExpertsModular` の任意の組み合わせについて、`FusedMoEModularKernel::forward()` の 1 回の呼び出しに対する Torch トレースを生成できます。
例: `python3 -m tests.kernels.moe.modular_kernel_tools.profile_modular_kernel --pf-type DeepEPLLPrepareAndFinalize --experts-type BatchedTritonExperts`

## FusedMoEPrepareAndFinalizeModular の実装 { #fusedmoeprepareandfinalizemodular-implementations }

利用可能なモジュラー prepare / finalize サブクラスの一覧は、[Fused MoE カーネルの機能](./moe_kernel_features.md#fused-moe-modular-all2all-backends)を参照してください。

## FusedMoEExpertsModular { #fusedmoeexpertsmodular_1 }

利用可能なモジュラー experts の一覧は、[Fused MoE カーネルの機能](./moe_kernel_features.md#fused-moe-experts-kernels)を参照してください。
