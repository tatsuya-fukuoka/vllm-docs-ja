# デュアルバッチオーバーラップ { #dual-batch-overlap }

## 動機 { #motivation }

vLLM における DBO（Dual Batch Overlap）の中心的な狙いは、MoE 層のスパースな all-to-all 通信を、その前後の計算とオーバーラップさせることです。現時点でこの仕組みは DP + EP のデプロイのみを対象としています。

## はじめに { #introduction }

デュアルバッチオーバーラップは、モデルランナーでバッチを分割し、2 つのワーカースレッドを作成して、それぞれのワーカースレッドでモデルを実行することで動作します。DBO が有効な場合、`FusedMoEModularKernel` 内の yield ポイントによって 2 つの CPU ワーカースレッド（UBatch スレッドとも呼びます）が交互に切り替わり、一方が計算を実行している間に、もう一方は通信を待つ形になります。コード全体を通じて、microbatch の短縮形として ubatch が使われることがあります。これは短縮形 µ-batch を ASCII で表したものです。

DBO の仕組みには `GpuModelRunner` と `ModularKernel` への変更が含まれ、`UBatchWrapper` と `UBatchContext` という 2 つのユーティリティクラスが定義されています。`UBatchWrapper` はスレッドのライフサイクルとモデルの CUDA graph 実行を管理します。`UBatchContext` は `ForwardContext` をラップし、2 つの UBatch スレッド間の同期を調整します。

現在 vLLM に実装されているオーバーラップのスケジュールは次のとおりです。

```python
# Schedule notation legend:
#    S = Shared expert
#    A0 = MLA qkv proj,
#    A1 = Core attn + out proj + MoE gate
#    D = Dispatch
#    C = Combine

# Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
# Order: D₁ send, A0₀, A1₀, D₁ recv, D₀ send, MLP₁, D₀ recv,
#        C₁ send, S₁, MLP₀, C₁ recv, C₀ send, S₀, A0₁, A1₁, C₀ recv.
# MLP_SHARED_OVERLAP = "mlp_shared_overlap"
```

## DBO を有効にして実行する { #running-with-dbo }

DBO を有効にするには、vllm serve コマンドに `--enable-dbo` 引数を渡します。これは `--data-parallel-size N`（N は 1 より大きい）および `--enable-expert-parallel` と併用する必要があります。さらに、2 つの設定つまみがあります。

* `--dbo-decode-token-threshold` そのバッチで DBO を有効にするために必要な、デコードのみのバッチにおける最小トークン数
* `--dbo-prefill-token-threshold` そのバッチで DBO を有効にするために必要な、プレフィルを 1 つ以上含むバッチにおける最小トークン数

現時点で DBO は DeepEP でのみサポートされているため、DeepEP をインストールしたうえで、`--all2all-backend` 引数を、ワークロードが主にデコードリクエストなら `deepep_low_latency`、主にプレフィルリクエストなら `deepep_high_throughput` に設定する必要があります。

次は、DP ランク 2 のサーバーをエキスパート並列と DBO 有効で起動するコマンドの例です。
例: `vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

`CUDA_VISIBLE_DEVICES` に少なくとも 2 台の GPU が見えている必要がある点に注意してください。

## DBO のコンポーネント { #dbo-components }

* GPUModelRunner
* UBatchWrapper
* UBatchContext

### GPU モデルランナー { #gpu-model-runner }

バッチは `GPUModelRunner` クラスによってマイクロバッチに分割されます。これは 2 つのステップで行われます。まず、マイクロバッチ化を適用するかどうかを判断するために、すべての DP ランク間で調整が行われます。マイクロバッチ化はすべての DP ランクで一様でなければなりません。いずれかの DP ランクでマイクロバッチ化が実行できない場合、すべてのランクで無効になります。すべての DP ランクがマイクロバッチ化を行う場合、合計トークン数は全ランク中の最大トークン数までパディングされます。パディング適用後に、いずれかのランクで 2 つ目のマイクロバッチが空になってしまう場合、マイクロバッチ化は中止され、どのランクもマイクロバッチ化を行いません。すべてのランクでマイクロバッチ化が開始されたら、2 つ目のステップが実行されます。`GPUModelRunner` が `CommonAttentionMetadata` を半分にスライスし、マイクロバッチごとに 1 つの attention メタデータが用意されます。

### UBatchWrapper { #ubatchwrapper }

gpu_ubatch_wrapper

`UBatchWrapper` クラスは、DBO のためのスレッド、UBatchContext、CUDA graph の管理をすべて担うモデルのラッパーです。GPU モデルランナーからは比較的透過的に見えるよう設計されています。

実装では、マイクロバッチごとに 1 回ずつ、モデルを合計 2 回実行します。各モデル呼び出しは UBatch スレッド内で行われます。これらのスレッドは並列に起動され、`UBatchContext` を使って同期されます。各スレッドには、自分が担当する半分のバッチを実行するためにスライスされた attention メタデータが渡されます。

DBO の CUDA graph は完全に `UBatchWrapper` が管理します。そのため、DBO は Full CUDA graph でのみ実行できます。ただし、DBO の CUDA graph がいったんキャプチャされれば、マルチスレッドや CPU の同期なしにリプレイできます。

#### インターフェース { #interfaces }

`__init__` メソッドは、モデル、VllmConfig、CUDAGraphMode、device を受け取ります。

`forward` メソッドはモデルの引数のみを受け取ります。`forward_context` に `ubatch_slices` オブジェクトが存在するかどうかで、DBO 付きで実行するかどうかを判断します。存在しない場合、モデルは DBO なしで実行されます。

### UBatchContext { #ubatchcontext }

ubatch_context

`UBatchContext` クラスは `ForwardContext` のラッパークラスで、`UBatchWrapper` クラスが 2 つの UBatch スレッドを同期するために使います。インスタンス化は `make_ubatch_contexts` を通じてのみ行うべきです。

いずれかの UBatch スレッドが `dbo_yield` の呼び出しに到達すると、そのスレッドは一時停止し、もう一方のスレッドを開始します。もう一方のスレッドは同じ `dbo_yield` の呼び出しに到達するまで実行されます。この「ピンポン」の動きは、モデルの実行が完了するまで、`dbo_yield` の呼び出しごとにスレッドを入れ替えながら続きます。

現在の実装では、`dbo_yield` と `dbo_maybe_run_recv_hook` の呼び出しはすべて `FusedMoEModularKernel.forward` メソッド内にあります。

#### インターフェース { #interfaces_1 }

`make_ubatch_context` 関数は、UBatch スレッドごとに 1 つずつ、2 つの `UBatchContexts` を初期化します。この関数は 2 つの CUDA ストリーム、既存の `ForwardContexts`、および CPU スレッドのバリアを受け取ります。`UBatchContexts` のインスタンス化にはこの関数のみを使うべきです。イベントの初期化はすべてこの関数が行います。

`dbo_register_recv_hook` メソッドは、もう一方の UBatch スレッドの `UBatchContext` において `FusedMoEPrepareAndFinalizeModular` クラスが返しうるコールバックを登録します。このコールバックは、もう一方のスレッドが `dbo_maybe_run_recv_hook` を呼び出したときに実行されます。通常は all-to-all カーネルの完了を待つために使われます。

`dbo_maybe_run_recv_hook` メソッドは、`dbo_register_recv_hook` 関数で設定されたコールバックが存在する場合に、それを実行します。

`dbo_yield` メソッドは、現在のスレッドをスリープさせ、もう一方の UBatch スレッドを起こします。
