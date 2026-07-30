# vLLM と torch.compile の統合をデバッグする方法 { #how-to-debug-the-vllm-torchcompile-integration }

要点:

- torch.compile のログを取得するには tlparse を使ってください。バグ報告やサポートの問い合わせにはこれらのログを添えてください。
- vLLM と torch.compile の統合は複数の要素からなります。vLLM は各要素を無効にするフラグを提供しています。

| オンラインのフラグ                    | オフラインのフラグ                                                                   | 効果                                               |
|--------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------|
| --enforce-eager                | enforce_eager=True                                                             | torch.compile と CUDA graph を無効にする                |
| -cc.mode=0                     | compilation_config=CompilationConfig(mode=CompilationMode.NONE)                | torch.compile のみを無効にする                          |
| -cc.mode=1                     | compilation_config=CompilationConfig(mode=CompilationMode.STOCK_TORCH_COMPILE) | torch.compile に対する vLLM-compile の改変を無効にする |
| -cc.cudagraph_mode=NONE        | compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE)        | CUDA graph のみを無効にする                            |
| -cc.backend=eager              | compilation_config=CompilationConfig(backend='eager')                          | TorchInductor を無効にする                               |
| -cc.ir_enable_torch_wrap=False | compilation_config=CompilationConfig(ir_enable_torch_wrap=False)               | vLLM IR のラッピングを無効にする                              |

## vLLM と torch.compile の概要 { #vllm-torchcompile-overview }

性能を高めるため、vLLM は torch.compile と CUDA graph を活用して処理を高速化します。torch.compile は PyTorch のコードに対して最適化されたカーネルを生成し、CUDA graph はオーバーヘッドを取り除きます。特に重要な点として、vLLM-compile は torch.compile そのものでは**ありません**。PyTorch Compile の内部 API を使って構築された独自のコンパイラです。

![vLLM-compile diagram](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/debug_vllm_compile/design_diagram.png)

- モデルが与えられると、バッチサイズ（トークン数）について動的な形で、TorchDynamo による全体グラフのキャプチャを行います。
- 続いて vLLM は、必要に応じてこのグラフを分割・特殊化し、TorchInductor を使って各グラフをコンパイル済みの成果物にコンパイルします。このステップでは、グラフをさらに最適化するために vLLM 独自の Inductor パスが使われることがあります。これにはディスパッチのオーバーヘッドを取り除く vLLM IR の lowering も含まれます。
- コンパイル済みの成果物は vLLM のコンパイルキャッシュに保存され、次回以降に読み込めるようになります。
- vLLM は CPU のオーバーヘッドを減らすために CUDA graph を適用します。

この 4 つのステップのいずれでも問題が起こりえます。問題が起きたときは、どのサブシステムが原因かを切り分けてみてください。そうすれば、性能への影響を最小限に抑えつつ信頼性を確保するために無効にすべきものを最小限にできますし、バグ報告の際に私たち（vLLM）にとっても助けになります。

設計の詳細については、次のリソースを参照してください。

- [Introduction to vLLM-torch.compile blogpost](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [vLLM-torch.compile integration design](./torch_compile.md)
- [vLLM IR design](./vllm_ir.md)
- [vLLM Office Hours #26](https://www.youtube.com/live/xLyxc7hxCJc?si=Xulo9pe53C6ywf0V&t=561)
- [Talk at PyTorch Conference 2025](https://youtu.be/1wV1ESbGrVQ?si=s1GqymUfwiwOrDTg&t=725)

## tlparse を使う { #use-tlparse }

torch.compile のログを見るには [tlparse](https://github.com/meta-pytorch/tlparse) を使ってください。これらのログには、torch.compile が生成する融合カーネルを含め、コンパイル過程のすべての段階が表示されます。

tlparse のインストール:

```sh
pip install tlparse
```

torch.compile のログを有効にするには、環境変数 `TORCH_TRACE=<dir>` を設定します。トレース中、そのディレクトリ内にランクごとのファイルが作成され、各ファイルにコンパイル中の成果物が記録されます。可能であれば、これらのログファイルをバグ報告に添えて送っていただけると非常に助かります。

使い方（オフライン推論）

```sh
TORCH_TRACE=~/trace_dir python my_script.py
tlparse ~/trace_dir/<rank_0_log_file>
```

使い方（サービング）

```sh
TORCH_TRACE=~/trace_dir vllm serve
# ctrl-c out of the server
tlparse ~/trace_dir/<rank_0_log_file>
```

ログファイルの 1 つを渡すと、`tlparse` コマンドはいくつかの HTML ファイルを出力します（たとえば `./tl_out/index.html` など）。それを開くとログを確認できます。次のような見た目になります。

![tlparse example](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/debug_vllm_compile/tlparse_inductor.png)

## vLLM と torch.compile の統合を無効にする { #turn-off-vllm-torchcompile-integration }

`--enforce-eager` を渡すと、vLLM と torch.compile の統合を無効にし、完全に eager モードで実行できます。これには CUDA graph の無効化も含まれます。

```sh
# Online
vllm serve --enforce-eager
```

```py
# Offline
LLM(model, enforce_eager=True)
```

torch.compile のみを無効にするには、コンパイル設定に `mode = NONE` を渡します（`-cc` は `--compilation_config` の短縮形です）。

```sh
# Online
vllm serve -cc.mode=0
```

```py
# Offline
from vllm.config.compilation import CompilationConfig, CompilationMode
LLM(model, compilation_config=CompilationConfig(mode=CompilationMode.NONE))
```

CUDA graph のみを無効にするには `cudagraph_mode = NONE` を渡します。

```sh
# Online
vllm serve -cc.cudagraph_mode=NONE
```

```py
# Offline
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
LLM(model, compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE))
```

vLLM IR は、functionalization、カスタムの融合、lowering といったコンパイルのパイプラインを多用します。これを無効にして vLLM IR の eager モードのディスパッチ挙動を捉えるには、`ir_enable_torch_wrap=False` を指定して実行してください。IR の torch wrap が既定で有効になるのは、`mode=VLLM_COMPILE` かつ `backend="inductor"`（既定）の場合のみです。

```sh
# Online
vllm serve -cc.ir_enable_torch_wrap=False
```

```py
# Offline
from vllm.config.compilation import CompilationConfig
LLM(model, compilation_config=CompilationConfig(ir_enable_torch_wrap=False))
```

## TorchDynamo のデバッグ { #debugging-torchdynamo }

vLLM は、モデルのコードが TorchDynamo（torch.compile のフロントエンド）によって全体グラフとしてキャプチャできることを要求します。TorchDynamo は Python のすべての機能をサポートしているわけではありません。ある機能をサポートできない場合、（fullgraph モードでは）エラーになります（これはグラフブレークと呼ばれることがあります）。

グラフブレークに遭遇した場合は、PyTorch の開発者が優先順位を付けられるよう、[pytorch/pytorch に issue を作成](https://github.com/pytorch/pytorch)してください。そのうえで、グラフブレークを避けるようコードを書き換えてみてください。詳細は [Dynamo のガイド](https://docs.pytorch.org/docs/stable/compile/programming_model.dynamo_core_concepts.html)を参照してください。

## 動的形状での全体グラフキャプチャのデバッグ { #debugging-dynamic-shape-full-graph-capture }

vLLM は、モデルの forward パスが、バッチサイズ（つまりトークン数）について動的な全体グラフとしてキャプチャできることを要求します。既定では、この 1 つのグラフを 1 つの成果物にコンパイルし、その成果物をすべてのバッチサイズで使います。

コードが動的形状でキャプチャできない場合、静かな不正な結果、明示的なエラー、あるいは CUDA の不正メモリアクセスが発生することがあります。たとえば次のコードは 1 つのグラフとしてキャプチャできません。

```py
if data.size[0] % 128 == 0:
    foo(...)
else:
    bar(...)
```

この問題は簡単に診断できます。tlparse を使い `compilation_metrics` をクリックしてください。バッチサイズに関するシンボリックな制約が表示されます。バッチサイズを制限する制約があれば、それが問題です。

![Bad tlparse example](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/debug_vllm_compile/dynamic_shapes.png)

これを避けるには、次のいずれかを行ってください。

1. トークン数にもとづく分岐を避ける
2. 分岐のロジックをカスタム演算子で包む。TorchDynamo はカスタム演算子の内部をトレースしません。

## 制約違反と動的形状の guard の問題のデバッグ { #debugging-constraint-violations-and-dynamic-shapes-guards-issues }

動的形状の guard は Dynamo の guard の一種です。これは、コンパイル済み成果物が有効であり続けることを保証するために、`torch.compile` が動的な次元（`seq_len` など）に付ける制約です。これらの guard は通常、フレームワークのコード、カスタムパス、あるいはユーザーのコードが動的形状の値にもとづいて分岐する際に現れます。

**例:**

```python
if x > 10:
    # path A
else:
    # path B
```

これにより、どちらの経路がトレースされたかに応じて `x > 10` または `x <= 10` の guard が作られます。

**vLLM の前提:**
vLLM は、torch.compile が追加するすべての guard は安全に破棄でき、コンパイル済みグラフを特定の入力形状に制約しないことを前提としています。この前提が破られると、ユーザーがデバッグしなければならない問題が生じることがあります。この前提が破られていることを示す兆候としては、実行時エラーや `ConstraintViolationError` があります。

動的な形状が単一の値に制約されると `ConstraintViolationError` が送出されます。制約違反のエラーに遭遇した場合や、動的形状の guard が誤って追加されている疑いがある場合は、より厳格な動的形状のモードを使って問題の切り分けに役立てられます。

```sh
# Online - using unbacked mode
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=unbacked

# Online - using backed_size_oblivious mode
vllm serve meta-llama/Llama-3.2-1B -cc.dynamic_shapes_config.type=backed_size_oblivious
```

```py
# Offline - using unbacked mode
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType
LLM(model, compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.UNBACKED)
))

# Offline - using backed_size_oblivious mode
from vllm.config.compilation import CompilationConfig, DynamicShapesConfig, DynamicShapesType
LLM(model, compilation_config=CompilationConfig(
    dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.BACKED_SIZE_OBLIVIOUS)
))
```

これらのモードはより厳格で、動的形状の guard の必要性を減らす、あるいはなくすため、問題の切り分けに役立ちます。

- `unbacked`: guard を許可しない unbacked symint を使うため、guard が誤って追加されている箇所を特定しやすくなります。
- `backed_size_oblivious`: guard についてより厳格なモードを使います。

動的形状のモードの詳細は、[動的形状と vLLM による guard の破棄](torch_compile.md#dynamic-shapes-and-vllm-guard-dropping)を参照してください。

### guard を表示する { #printing-guards }

コンパイル中に追加されるすべての guard を確認するには `TORCH_LOGS=+dynamic` を使います。

```sh
TORCH_LOGS=+dynamic vllm serve meta-llama/Llama-3.2-1B
```

ログの中で `[guard added]` を探すと、guard がどこで追加されているかが分かります。これにより、どの演算が guard の誤った追加を引き起こしているかを特定できます。

## TorchInductor のデバッグ { #debugging-torchinductor }

TorchInductor はキャプチャされたグラフを受け取り、1 つ以上の Triton カーネルを呼び出す Python のコードへコンパイルします。まれに（そして残念なことに）、誤った Triton カーネルを生成することがあります。これは、静かな不正な結果、CUDA の不正メモリアクセス、あるいは明示的なエラーとして現れることがあります。

### Inductor の実行時アサーション { #inductor-runtime-assertions }

既定では（torch 2.12 未満の場合）、vLLM は Inductor の実行時アサーション（`assert_size_stride`、`assert_alignment`）を無効にしています。大きなモデルでは forward パスあたり約 2 ミリ秒のオーバーヘッドが生じるためです。`VLLM_LOGGING_LEVEL=DEBUG` を設定すると自動的に再度有効になり、デバッグ時には形状 / stride の完全な検証が行われます。

```sh
VLLM_LOGGING_LEVEL=DEBUG vllm serve <model>
```

`--compilation-config` で明示的に上書きすることもできます。

```sh
vllm serve <model> -cc.inductor_compile_config='{"size_asserts": true, "alignment_asserts": true, "scalar_asserts": true}'
```

torch 2.12 以降では、PyTorch は効率的な「一度だけアサートする」戦略を採用しており、vLLM がこれらのフラグを抑制することはなくなりました。

TorchInductor が原因かどうかを調べるには、コンパイル設定に `backend='eager'` を渡して無効にできます。

```sh
# online
vllm serve -cc.backend=eager
```

```py
# offline
LLM(compilation_config=CompilationConfig(backend='eager'))
```

Inductor が原因であれば、[PyTorch にバグを報告](https://github.com/pytorch/pytorch)してください。挑戦してみたい場合は、（tlparse で場所を特定できる）Inductor の出力コード内の Triton カーネルをデバッグすることもできます。

![tlparse example](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/debug_vllm_compile/tlparse_inductor.png)

Inductor の出力コードを表示するには `TORCH_LOGS=output_code <command>` も使えます。

### TorchInductor のコードを編集可能にする { #editable-torchinductor-code }

`VLLM_COMPILE_CACHE_SAVE_FORMAT=unpacked` を設定するか `-cc.compile_cache_save_format=unpacked` を渡すことで、実行される TorchInductor のコードを編集できます。既定は `binary` で、この場合は編集できません。

これは有用なテクニックです。出力コードにブレークポイント（`torch.distributed.breakpoint()` など）や print 文を入れられます。

## vLLM-compile のキャッシュのデバッグ { #debugging-vllm-compile-cache }

vLLM は torch.compile の成果物のために独自のキャッシュを構築しています。成果物を一度コンパイルすれば、以降は再利用できるという考え方です。これは [torch.compile のコンパイラキャッシュ](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)の上に載る層です。

torch.compile のコンパイラキャッシュは非常に安定していますが、残念ながら vLLM のコンパイラキャッシュは常に正しいとは限りません。`VLLM_DISABLE_COMPILE_CACHE=1` を設定すると無効にできます。

このキャッシュは手動で削除することもできます。

- vLLM のコンパイルキャッシュは `rm -rf ~/.cache/vllm` で削除します（場所が変わっていないかログを確認してください）
- torch.compile の組み込みキャッシュは `rm -rf /tmp/torchinductor_$(whoami)` で削除します

vLLM のキャッシュは、キャッシュキーからコンパイル済み成果物へのマッピングです。vLLM は複数の要素（設定フラグやモデル名など）を組み合わせてキャッシュキーを計算します。vLLM のコンパイルキャッシュが誤っている場合、通常はいずれかの要素が抜けていることを意味します。vLLM がキャッシュキーの一部をどう計算しているかは[この例](https://github.com/vllm-project/vllm/blob/18b39828d90413d05d770dfd2e2f48304f4ca0eb/vllm/config/model.py#L310)を参照してください。

vLLM のコンパイルキャッシュは、コンパイル対象のコードが最終的にシリアライズ可能であることを要求します。そうでない場合、保存時にエラーになります。通常の対処は次のいずれかです。

- シリアライズできない部分を書き換える（現時点では何がシリアライズ可能かを判断しにくいため、難しい場合があります）
- バグを報告する
- `VLLM_DISABLE_COMPILE_CACHE=1` を設定してエラーを無視する（ただし、ウォーム状態でのサーバー起動がかなり遅くなります）

## CUDA graph のデバッグ { #debugging-cudagraphs }

CUDA graph は次のことを可能にする機能です。

- 1 つ以上の CUDA カーネルを起動する呼び出し可能オブジェクトを CUDA graph としてキャプチャする
- その CUDA graph をリプレイする

キャプチャされた CUDA graph には、キャプチャ処理中に使われたすべてのメモリが含まれます。CUDA graph のリプレイは、まったく同じメモリ領域を読み書きします。

このことから、いくつかの制約が生じます。

1. 新しいデータに対して CUDA graph を使うには、そのデータを CUDA graph が読み取るバッファへコピーする必要があります。
2. CUDA graph がキャプチャするのは CUDA カーネルのみで、CPU 上の処理はキャプチャしません。

vLLM は生の CUDA graph API を使っており、誤った使い方をすると安全ではありません。

CUDA graph のみを無効にするには `cudagraph_mode = NONE` を渡します。

```sh
# Online
vllm serve -cc.cudagraph_mode=NONE
```

```py
# Offline
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
LLM(model, compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE))
```
