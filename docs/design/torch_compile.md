# `torch.compile` の統合 { #torchcompile-integration }

vLLM の V1 アーキテクチャでは `torch.compile` が既定で有効になっており、フレームワークの重要な構成要素です。このドキュメントでは、`torch.compile` の使われ方を理解するための簡単な例を順を追って説明します。

例を通して、一般的な Llama モデルを実行し、詳細をすべて表示するためにデバッグレベルのログを有効にします。使用するコマンドは `VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B` です。

!!! note
    `torch.compile` の統合に関する詳細と最新の進捗については、この[ブログ記事](https://blog.vllm.ai/2025/08/20/torch-compile.html)を参照してください。

## コンパイルキャッシュ { #compilation-cache }

非常に詳細なログには、次のような行が見られます。

```console
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```

vLLM は利用可能なすべての要素を考慮して、コンパイル成果物を保存するディレクトリを決定します。つまり、デプロイ時に `~/.cache/vllm/torch_compile_cache` ディレクトリ全体をそのままコピーすれば、大幅にコンパイル時間を節約でき、vLLM インスタンスの起動を高速化できます。

考慮される要素は次のとおりです。

- 関連するすべての設定（[config フォルダ](../../vllm/config)内の各設定にある `compute_hash` 関数を参照）
- PyTorch の設定（[compiler_interface.py](../../vllm/compilation/compiler_interface.py) の `compute_hash` 関数を参照）
- モデルの forward 関数と、forward 関数から呼ばれる関連関数（下記参照）

これらの要素をすべて考慮しているため、通常はキャッシュを安全に利用でき、予期しない挙動が起きることはありません。そのため、キャッシュは既定で有効です。コンパイル処理をデバッグしたい場合や、キャッシュが何らかの問題を引き起こしていると疑われる場合は、環境変数 `VLLM_DISABLE_COMPILE_CACHE=1` を設定して無効にできます。

vLLM の `torch.compile` 統合に特有の点として、リクエストを受け付ける前にすべてのコンパイルが完了することが保証されています。リクエストが新たなコンパイルを引き起こすことはありません。そうでなければ、エンジンがそのリクエストでブロックされ、応答時間に予期しないスパイクが生じてしまいます。

既定では、キャッシュはコンパイル成果物をバイナリファイルとして保存します。デバッグのために生成されたコードを直接扱いたい場合は、コンパイル設定の `compile_cache_save_format=unpacked` フィールドを設定するか、これを省略して環境変数 `VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked` を設定してください。

## 動的形状と vLLM における guard の削除 { #dynamic-shapes-and-vllm-guard-dropping }

`torch.compile` は、必要に応じてためらいなく動的形状に guard をかけるよう設計されています。これは、そうした guard の多くが実質的な意味を持ちうるにもかかわらず guard を削除する、という vLLM の `torch.compile` の方針と相反します。

`torch.compile` は 2 種類の動的形状、`backed` と `unbacked` を提供します。
`torch.compile` は `backed` の動的形状に guard をかけ、guard が追加されないことを保証しません。ユーザーコード、dynamo、inductor、autograd のいずれもが guard を追加し得ます。さらに 0/1 特殊化については、その範囲で分岐に遭遇していなくても、backed のシンボルは無条件に 0、1、あるいは 2 以上へ特殊化されます。

一方 `unbacked` の動的形状は、guard がかけられないことが保証され、0/1 特殊化もされません。ただし、その値を必要とする分岐に遭遇し、unbacked に対する明示的な処理が定義されていない場合、データ依存エラー（DDE）が送出される可能性があります。フレームワークは、DDE を送出せずに汎用的な経路を選ぶ方向へ収束しつつあります。unbacked を使う欠点としては、性能上のバグや汎用経路の選択によって最適化の機会を逃すこと、また例に依らない固定のヒントを使うこと（これは override_hint API により近く修正される予定です）が挙げられます。汎用経路を選ぶ例としては、シンボリックに証明できない場合に contiguous() や reshape() を呼ぶ関数で入力が連続でないと仮定し、clone を導入する変更を行うことが挙げられます。

`backed_size_oblivious` は、unbacked に対する明示的な処理が定義されている箇所で backed のシンボルを unbacked として扱うためのフラグです。このモードでは、フレームワークのコード内で 0/1 特殊化がおおむね回避され、既定の 0/1 特殊化も行われません。ただし、とくにユーザーコードやカスタムパスに起因して torch.compile が guard をかけないという保証は依然としてありません。`backed_size_oblivious` は PyTorch compile では実験的であり、非推奨になる可能性があります。とはいえ `backed` より安全な選択肢であり、性能が落ちる確率は `unbacked` より低くなります。

### 動的形状の設定 { #configuring-dynamic-shapes }

`DynamicShapesConfig` の `type` フィールドを設定することで、動的形状の挙動を制御できます。`BACKED`（既定）、`UNBACKED`、`BACKED_SIZE_OBLIVIOUS` の 3 つのモードから選べます。

#### オフライン推論の例（LLM クラスを使う場合） { #offline-inference-example-using-llm-class }

オフライン推論で `LLM` クラスを使う場合、動的形状は `compilation_config` パラメータで設定できます。

```python
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType

# Example: Using backed_size_oblivious (experimental, safer than backed)
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS
        )
    )
)

# Example: Using unbacked (strongest guarantee against guards)
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    compilation_config=CompilationConfig(
        dynamic_shapes_config=DynamicShapesConfig(
            type=DynamicShapesType.UNBACKED
        )
    )
)

# Generate outputs
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

#### オンラインサービングの例（vllm serve を使う場合） { #online-serving-example-using-vllm-serve }

オンラインサービングで `vllm serve` を使う場合、動的形状は `--compilation-config` フラグで設定できます。

```bash
# Example: Using unbacked
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"dynamic_shapes_config": {"type": "unbacked"}}'


# Alternative: Using dot notation (simpler for single values)
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked
```

#### 適切なモードの選び方 { #choosing-the-right-mode }

- **BACKED**（既定）: 最大の性能を得るために、安全でない可能性のある guard の削除を受け入れられる場合に使います。guard が不健全に追加されたうえで無視されることがあります。

- **UNBACKED**: guard に対する最も強い保証が必要な場合に使います。最も保守的な選択肢ですが、一部の最適化の機会を逃す可能性があります。

- **BACKED_SIZE_OBLIVIOUS**: guard の回避と性能のバランスを取りたい場合に使います。この実験的なモードは BACKED より安全ですが、UNBACKED ほど保守的ではありません。

## Python コードのコンパイル { #python-code-compilation }

非常に詳細なログには、次のような行が見られます。

??? console "ログ"

      ```text
      DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

      DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
      DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

      DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
      DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
      ```

これは Python コードのコンパイル、すなわち Dynamo によるグラフキャプチャに関するものです。`xxx/vllm/model_executor/models/llama.py:339` のコードを持つ関数、つまりコンパイル対象モデルの `forward` 関数のトレースを試みます。forward pass の途中では、ログに示されているように、Dynamo によって呼び出され、インライン化される他の関数もあります。これには `xxx/torch/nn/modules/module.py` の PyTorch の関数（PyTorch の `nn.Module` が使うもので、モジュールの属性アクセスが関数呼び出しを引き起こすため）や、vLLM の通信 / attention / 活性化関数などが含まれます。トレースされたファイルはすべて、使用するキャッシュディレクトリの決定時に考慮されます。したがって、これらのファイルのコードを変更するとコンパイルキャッシュがミスし、再コンパイルが行われます。

Dynamo によるコンパイルの結果は、`~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py` に保存される新しい関数です。通常この関数は、モジュールからテンソルを取り出し、トレースされた計算グラフに渡します。計算グラフは `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` に保存されます。

## 計算グラフの処理 { #computation-graph-processing }

計算グラフには、すべてのテンソルに形状の注釈が付いています。入力は input id、position id、モデルの重みとバッファで、出力は最終的な hidden states です。lm head の射影とサンプリング処理はグラフに含まれない点に注意してください。

計算グラフへの入力のほとんどは静的な形状を持ちます。モデルの重みとバッファであり、モデルの生存期間中に変化しないためです。シンボリックな形状を持つのは input id と position id だけで、これらはバッチごとに形状が変わり得ます。ただし、両者は同じシンボリック形状を共有します。つまり、計算グラフにおいて変化するサイズはバッチサイズ（現在の forward pass で処理されるトークン数）だけです。

attention の処理は複雑で、複雑な形状を持つ KV キャッシュとやり取りする必要があります。幸いなことに、attention の出力は attention の入力 query と同じ形状を持ちます。そこで attention の処理全体を PyTorch のカスタム op `torch.ops.vllm.unified_attention_with_output` にラップし、Dynamo が内部の処理を検査しないようにしています。こうすることで、attention の処理は複雑であっても、Dynamo から見ればモデルの計算グラフをフルグラフとしてキャプチャできます。

計算グラフはさらに `splitting_ops`（通常は attention の処理）によって断片に分割されます。そのため `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` ファイルには多数のサブモジュールが見られ、各サブモジュールが分割後のグラフの断片に相当します。

- attention の処理自体が 1 つのサブモジュールです。
- ある attention から次の attention までの計算グラフの部分が 1 つのサブモジュールです。

各サブモジュールはインデックスで識別でき、個別に処理されます。

## 計算グラフのコンパイル { #computation-graph-compilation }

非常に詳細なログには、次のような行も見られます。

```console
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

これは、計算グラフの最初の断片（シンボリック形状を表す形状 `None`）が Inductor によってコンパイルされたこと（キーは `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`）を意味します。コンパイル済みカーネルは `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py` に保存されます。このファイルを開けば、Inductor が最終的に実行するコードを確認できます。

もう 1 つの細かい点として、1 番目のグラフと 15 番目のグラフは同じキーを持ちますが、0 番目と 16 番目のグラフは異なることが分かります。これは想定どおりです。グラフを attention op で分割するため、次の 3 種類のサブグラフが得られます。

- attention より前の最初の層
- ある attention から次の attention までの中間の各層
- attention より後の最後の層

すでにキャッシュディレクトリが存在する場合（同じコードを 2 回目に実行する場合など）、次のようなログが表示されます。

```console
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

この場合、Inductor によるコンパイルは完全にスキップされ、前回得られたコンパイル成果物をディスクから読み込みます。

上記の例では、Inductor は汎用的な形状（すなわちシンボリック形状）向けにコンパイルしているだけです。特定の形状向けにコンパイルさせることもできます。例:

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

この場合、バッチサイズ `1, 2, 4, 8` 専用のカーネルもコンパイルされます。このとき、計算グラフ内のすべての形状が静的かつ既知になるため、最大の性能を目指してオートチューニングが有効になります。初回の実行は時間がかかることがありますが、次回以降はチューニングを省略して、チューニング済みのカーネルを直接実行できます。

すべての形状が既知であれば、`torch.compile` は複数の設定を比較でき、カーネルを実行するためのより良い設定を見つけられることがよくあります。たとえば、次のようなログが表示されます。

??? console "ログ"

    ```
    AUTOTUNE mm(8x2048, 2048x3072)
      triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
      mm 0.0160 ms 81.6%
      triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
      triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
      triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
      triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
      triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
      triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
    SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
    ```

これは、形状 `8x2048x3072` の行列積に対して `torch.compile` がさまざまな設定の triton テンプレートを試し、既定のコード（cublas ライブラリへディスパッチするもの）よりはるかに高速になったことを意味します。

残念ながらオートチューニングにはかなりの時間がかかるため（モデルサイズとバッチサイズにもよりますが、数秒から数分）、後で使えるようキャッシュできるとはいえ、使いやすさを優先して既定では無効にしています。最大の性能を求める場合は、特定の形状をコンパイルして試してみることを推奨します。

## CUDA graph のキャプチャ { #cudagraph-capture }

vLLM の V1 アーキテクチャは、区分的（piecewise）コンパイルに合わせた区分的 CUDA graph を使います。計算グラフ全体は前述のように分割され、attention のあいだのグラフ断片（最初の attention より前のグラフと、すべての attention より後の最後のグラフを含む）についてのみ CUDA graph をキャプチャします。これは一般的な観察にもとづいています。attention 間の計算は通常トークン単位で、CUDA graph にとって扱いやすい一方、attention の処理を CUDA graph 互換にするのは簡単ではありません。そこで attention を eager モードで実行し、それ以外の処理を CUDA graph 内で実行することで、attention の柔軟性を保っています。

区分的 CUDA graph には、きめ細かいメモリ管理も備わっています。その目的は、CUDA graph から attention カーネルだけを除外し、それ以外のモジュールとメモリ確保処理はすべて CUDA graph 内に保つことです。V1 で attention の処理が出力テンソルを attention の入力として受け取っているのは、このためです。

CUDA graph はコンパイラのバックエンドがキャプチャ・管理し、対応する CUDA graph がキャプチャ済みのバッチサイズのときに再生されます。モデルの呼び出し側（model runner）は、入力バッファを正しく管理することだけを保証すればよく、中間バッファはすべてコンパイラのバックエンドが自動的に管理します。

既定では、vLLM は CUDA graph をキャプチャするサイズの集合を自動的に決定します。設定 `cudagraph_capture_sizes` で上書きすることもできます。

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

この場合、指定したサイズについてのみ CUDA graph がキャプチャされます。CUDA graph のキャプチャを細かく制御したい場合に便利です。

### フル CUDA graph のキャプチャ { #full-cudagraph-capture }

CUDA graph 互換の attention バックエンドを使っている場合は、attention も CUDA graph に含められます。小さめのモデルや MoE のデコード速度など、場合によっては性能が向上します。詳細は [CUDA Graphs](cuda_graphs.md) を参照してください。
