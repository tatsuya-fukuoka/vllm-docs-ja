# CUDA Graphs { #cuda-graphs }

この文書では、これまでの [torch.compile 統合](torch_compile.md)に加えて導入された vLLM v1 の新しい CUDA Graphs モードを紹介します。要点は次のとおりです。

1. 柔軟な `cudagraph_mode` 設定を追加した
2. フル CUDA Graphs のサポートをコンパイルと直交させた
3. バッチごとに適切な実行時モードと CUDA Graphs を自動選択する中央コントローラとして、CUDA Graphs のディスパッチャを導入した

このドキュメントでは、次の内容を扱います。

* [動機](#motivation)
* [CUDA Graphs のモード](#cudagraphmodes)
* [詳細設計](#detailed-design)
* [各 CUDA Graphs モードの使用例](#usage-guide)
* [Vision Encoder（ViT）の CUDA Graphs](cuda_graphs_multimodal.md)

!!! note
    このドキュメントでは、純粋なデコード（`max_query_len=1`）や投機的デコード（`max_query_len =1+num_spec_tokens`）のバッチを **uniform decode** バッチと呼び、その反対を **non-uniform** バッチ（すなわちプレフィル、あるいはプレフィルとデコードが混在したバッチ）と呼びます。

!!! note
    以下の内容は、主に <https://github.com/vllm-project/vllm/pull/20059> の最後のコミットにもとづいています。

## 動機 { #motivation }

当初の区分的（piecewise）コンパイルは、CUDA graph 非対応の処理（主に attention）を除外したうえで区分的な CUDA graph のキャプチャを可能にするために作られました。これにより、すべての attention バックエンドとの互換性を保ちながら CUDA graph による高速化をある程度得られました。その後、区分的にコンパイルしないことで「フル CUDA graph」のサポートを追加し、attention が CUDA graph に対応している場合にさらにレイテンシを削減できるようにしました。しかし、コンパイルと CUDA graph キャプチャがこのように密結合していたため、柔軟性に乏しい「全か無か」の体験になっていました。また多くの attention バックエンドは、統一的な「フル」CUDA Graphs キャプチャに対応していない（現時点で対応しているのは FlashAttention 3 のみ）か、純粋なデコードバッチについてのみ CUDA Graphs をサポートしています（Flashinfer、FlashMLA、Mamba など）。その結果、性能と互換性のトレードオフが分かりにくくなり、CUDA Graphs のサポート状況も一貫せず、コード構造も次第に複雑になっていました。

そこで、次の特徴を備えた、よりきめ細かい CUDA Graphs の仕組みを目指すことにしました。

* プレフィル / 混在バッチと（uniform な）デコードバッチを CUDA Graphs 上で明示的に区別し、別々にキャプチャする。
* 機能の直交性のため、CUDA graph のキャプチャロジックをコンパイルから（可能な限り）分離する。具体的には次を意味します。
    * 同じコンパイル済みグラフを使って区分的 CUDA graph とフル CUDA graph の両方をキャプチャする
    * コンパイルなしでフル CUDA graph をキャプチャする
* バッチの構成に応じて、実行時にフル CUDA graph と区分的 CUDA graph を切り替える。
* コードの複雑さを減らし拡張しやすくするため、CUDA graph の挙動を中央で制御する。

これらの特徴により、起動時間と性能のさまざまなトレードオフや機能サポートに対して、CUDA graph のキャプチャとコンパイルを最大限柔軟に構成できます。

## `CudagraphModes` { #cudagraphmodes }

[`CUDAGraphMode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/compilation/#vllm.config.compilation.CUDAGraphMode) は、`CompilationConfig.cudagraph_mode` で調整する唯一のつまみです。

* `NONE` — CUDA Graphs を無効にします。デバッグに適しています。
* `PIECEWISE` — 単一モードの戦略です（かつての既定値）。最も柔軟で、attention など CUDA Graphs 非対応の処理は eager のまま、それ以外を CUDA Graphs に載せます。区分的コンパイルが必要です。
* `FULL` — 単一モードの戦略で、non-uniform バッチについてのみフル CUDA Graphs をキャプチャします。uniform decode のバッチは互換性があるため、同じ batch_size の non-uniform バッチの CUDA Graph を再利用します。小さなモデルや短いプロンプトのワークロードに適しています。
* `FULL_DECODE_ONLY` — uniform decode についてフル CUDA Graph を使い、プレフィル / 混在などには CUDA graph を使いません。プレフィルの重要度が低い P/D 構成のデコードインスタンスに適しており、`PIECEWISE` CUDA Graphs に必要なメモリを節約できます。
* `FULL_AND_PIECEWISE` —（既定のモード）uniform decode にはフル CUDA Graph を、それ以外には区分的 CUDA Graphs を使います。一般に最も高性能な設定で、とくに小さなモデルや MoE の低レイテンシ用途に向きますが、メモリを最も多く必要とし、キャプチャにも最も時間がかかります。

既定値: 区分的コンパイルを伴う v1 では、より高い性能のため `FULL_AND_PIECEWISE` が既定になります（プーリングモデルでは引き続き `PIECEWISE`）。それ以外の場合、たとえば区分的コンパイルが利用できない場合は `NONE` が既定になります。

`NONE`、`PIECEWISE`、`FULL` は単一モードの設定であり、それぞれ従来の eager 実行、区分的 CUDA Graphs、フル CUDA Graphs の実装に相当します。一方 `FULL_DECODE_ONLY` と `FULL_AND_PIECEWISE` は新たに追加された二重モードの設定で、実行時のバッチに応じて具体的な実行時モードを動的に切り替えるディスパッチが必要になります。

!!! note
    ここでは、単一モードの `NONE`、`PIECEWISE`、`FULL` を CUDA Graphs ディスパッチにおける実行時モードとして扱います。二重モードを使う場合、ディスパッチャはバッチの構成に応じて、常にその構成モードのいずれか（適切な CUDA Graph がない場合は `NONE`）へディスパッチします。

カスケード attention は CUDA graph 非対応ですが、現在はすべての CUDA graph モード設定と共存できます。バッチがカスケード attention を使う場合、利用可能であれば常に `PIECEWISE` モードへ（そうでなければ `NONE` へ）ディスパッチされます。

!!! note
    すべての CUDA Graph モードがすべての attention バックエンドと互換なわけではありません。vLLM は、最も近いサポート済みモードへ自動的に「ダウングレード」します。たとえば、あるバックエンドが純粋なデコード / uniform バッチについてのみ CUDA Graphs をサポートする場合、区分的コンパイルが有効なら `FULL` を `FULL_AND_PIECEWISE` に、そうでなければ `FULL_DECODE_ONLY` に変換します。

## 詳細設計 { #detailed-design }

### 概要 { #overview }

新しい CUDA Graphs のロジックは区分的コンパイルの上に構築され、CUDA Graphs の実行時モードの二重切り替えをサポートします。システムは次の中核コンポーネントから成ります。

* [`CUDAGraphWrapper`](https://docs.vllm.ai/en/v0.26.0/api/vllm/compilation/cuda_graph/#vllm.compilation.cuda_graph.CUDAGraphWrapper): ラップした callable に対する CUDA graph のキャプチャと再生を扱うラッパー
* [`CudagraphDispatcher`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/cudagraph_dispatcher/#vllm.v1.cudagraph_dispatcher.CudagraphDispatcher): CUDA Graphs に関する唯一の信頼できる情報源を保持し、それらの間のディスパッチを扱う中央コントローラ
* [`CUDAGraphMode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/compilation/#vllm.config.compilation.CUDAGraphMode): サポートされるモードと実行時モードを表す列挙型（上記で紹介）
* [`BatchDescriptor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/forward_context/#vllm.forward_context.BatchDescriptor): ディスパッチに使う、実行時バッチの一意な表現

inductor コンパイルを伴う CUDA Graphs について、従来と現在の設計パターンの比較を次の図に示します。従来は CUDA Graphs のロジックとコンパイルのロジックが vLLM の `PiecewiseBackend` に密結合しており、CUDA Graphs は `batch_size` によって暗黙にディスパッチされていました。現在は CUDA Graphs のロジックが `CUDAGraphWrapper` クラスに分離され、フルと区分的の双方の CUDA Graphs の能力を担い、ディスパッチは `CudagraphDispatcher` を通じて**実行時モード**と**ディスパッチキー**としての `BatchDescriptor` により**明示的に**行われます。

**変更前:**

![previous_design](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/cuda_graphs/previous_design.png)

**変更後:**

![new_design](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/cuda_graphs/current_design.png)

### `BatchDescriptor` { #batchdescriptor }

[`BatchDescriptor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/forward_context/#vllm.forward_context.BatchDescriptor) は、CUDA Graphs の実行時モードと並んで `ForwardContext` に含まれるコンポーネントで、実行時のディスパッチキーの中核となる構造です。定義は次のとおりです。

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int
    uniform: bool = False
    has_lora: bool = False
```

ここで `num_tokens` はパディング後のトークン長になり得ます。`uniform` は、すべてのリクエストが同じ query 長を持つかどうかを示します。多くの attention バックエンドは、バッチが uniform の場合にのみフル CUDA graph をサポートします。純粋なデコードバッチは uniform ですが、query 長が 1（すなわち `num_tokens == num_reqs`）とは限りません。これは投機的デコードの検証パスで起こり、そこでは「デコード」バッチの query 長が `1+num_spec_tokens` になります。

この構造の目的は、CUDA Graphs の 1 項目に対応する（パディング後の）バッチを、可能な限り少ない要素で一意に識別することです。

!!! note
    `BatchDescriptor` の定義は、今後より一般的な状況に対応するため拡張される可能性があります。たとえば、複数の異なる uniform decode 長の設定をサポートするために `uniform_query_len` のような項目を追加する（<https://github.com/vllm-project/vllm/pull/23679>）ことや、入力が必ずしもトークン長に依存しないモデル（一部のマルチモーダル入力など）で CUDA Graphs をサポートするための変更などが考えられます。

### `CudagraphDispatcher` { #cudagraphdispatcher }

[`CudagraphDispatcher`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/cudagraph_dispatcher/#vllm.v1.cudagraph_dispatcher.CudagraphDispatcher) は、有効なディスパッチキーの集合を 2 つ（`FULL` 実行時モード用と `PIECEWISE` 実行時モード用）維持し、モデルの forward を実行する前に正しい実行時モードとディスパッチキーへディスパッチする役割を担います。初期キー（パディング後の入力に対する大まかな batch_descriptor）を受け取り、選択した実行時モードと最終的な batch_descriptor を返し、その決定を forward context を通じて CUDAGraphWrapper のインスタンスに伝えます。利用可能な CUDA Graph のキーについては `CudagraphDispatcher` が唯一の信頼できる情報源であり、`CUDAGraphWrapper` のインスタンスは、どの CUDA Graphs へディスパッチするかについて forward context を無条件に信頼できます。これによりラッパー側のコードを簡素化し、ロジックをディスパッチャに集約できます。

ディスパッチキーは、ディスパッチャの `initialize_cudagraph_keys` メソッドで初期化されます。このメソッドは、想定されるすべての attention バックエンドの初期化後に gpu_model_runner から呼ばれます。将来的には、ここでさまざまな CUDA Graphs の組み合わせを「用意」する、より高度な処理も可能になります。現時点では、コンパイル設定の `cudagraph_mode` の `decode_mode` / `mixed_mode` の有効な組み合わせと `cudagraph_capture_sizes` にもとづいて、利用可能なキーを追加するだけです。

ディスパッチのコードは次のようになります。

```python
batch_descriptor=BatchDescriptor(num_tokens=num_input_tokens, uniform_decode=...)
runtime_mode, batch_descriptor = cudagraphdispatcher.dispatch(batch_descriptor)
# execution
with set_forward_context(
    ..., 
    cudagraph_runtime_mode=runtime_mode, 
    batch_descriptor=batch_descriptor,
):
     output = self.model(...)
```

`dispatch()` メソッドの内部では、ディスパッチャが適切な CUDA Graphs の実行時モードと既存のディスパッチキーを探して返します。基本的には `FULL` > `PIECEWISE` > `None` の優先順位で既存のキーを探索します。該当するディスパッチキーがなければ、既定で eager 実行のための `NONE` モードを返します。実装は[こちら](https://github.com/vllm-project/vllm/blob/main/vllm/v1/cudagraph_dispatcher.py#L91)にあります。

モデル実行時のワークフローを簡略化して示すと次のようになります。
![executor_runtime](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/cuda_graphs/executor_runtime.png)

### `CUDAGraphWrapper` { #cudagraphwrapper }

[`CUDAGraphWrapper`](https://docs.vllm.ai/en/v0.26.0/api/vllm/compilation/cuda_graph/#vllm.compilation.cuda_graph.CUDAGraphWrapper) のインスタンスは runnable をラップし、CUDA Graphs の機能を付加したうえでその runnable と同じように振る舞います。各ラッパーのインスタンスは特定の `runtime_mode`（`PIECEWISE` または `FULL` に限られます）に紐づき、キャプチャ / 再生と、runnable のパススルー（直接呼び出し）を担当します。実行時、各ラッパーは次のように動作します。

1. グローバルな forward context から runtime_mode と batch_descriptor（ディスパッチキー）を確認する。
2. runtime_mode が `NONE` であるか、ラッパー自身のモードと一致しない場合は、runnable を直接呼び出す。
3. それ以外、すなわち runtime_mode がラッパーのモードと一致する場合、ラッパーは CUDA Graphs のキャプチャ（キーが存在しなければ新しいエントリを作成してキャッシュする）または再生（キーがキャッシュに存在する場合）を行う。

上記の手順は、CUDA Graphs のラッパーが forward context の内容（ディスパッチャが制御）をそのまま信頼するという前提にもとづいています。これによりロジックを簡素化・集約でき、複雑さだけでなく、ラッパーとディスパッチャの状態が食い違うリスクも減らせます。また、`FULL` と `PIECEWISE` の両方の実行時モードで同じラッパークラスを再利用できます。実装は[こちら](https://github.com/vllm-project/vllm/blob/f751e50b7a2aae3110d83ed0d88202fc91b3e78a/vllm/compilation/cuda_graph.py#L106)を参照してください。

#### ネストされたラッパーの設計 { #nested-wrapper-design }

フル CUDA Graphs と区分的 CUDA Graphs を共存・両立させる中核的な仕組みが、ネストされた CUDA Graphs ラッパーの設計です。これは、単一の区分的 FX グラフのみを用いる区分的コンパイルの上に構築されています。フル CUDA Graphs の機能のためにモデル全体を FULL モードのラッパーで包み、同時に各区分的バックエンドをコンパイル内部で `PIECEWISE` モードのラッパーで包みます。

以下のフローチャートが、その動作を分かりやすく示しています。
![wrapper_flow](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/cuda_graphs/wrapper_flow.png)

したがって `FULL` 実行時モードでは、区分的ラッパーが有効にならないため、フル CUDA Graph のキャプチャ / 再生を安全に行えます。`PIECEWISE` モードでも同様で、`FULL` モードのラッパーと `PIECEWISE` モードのラッパーのあいだに衝突はありません。`NONE` 実行時モードでは `FULL` と `PIECEWISE` のどちらのラッパーも有効にならないため、そのまま eager 実行になります。

### フル CUDA Graph のキャプチャとウォームアップ { #full-cuda-graph-capturing-warm-up }

CUDA Graphs のキャプチャは、runner が `NONE` 以外の実行時モードでモデルの forward を最初に呼び出したとき（`_dummy_run` を使用）に行われます。フル CUDA Graph のキャプチャでは、attention のメタデータを適切に設定して、下位の attention バックエンドが意図したカーネルのルーチンを起動するようにし、それぞれのケース（プレフィル / 混在バッチ、uniform_decode バッチ）を明示的にキャプチャします。プレフィル / 混在バッチと uniform_decode バッチを区別するうえで最も重要なのは、attn_metadata の `max_query_len` です（ほとんどの attention バックエンドで当てはまります）。uniform_decode では意図した `uniform_query_len` を設定し、そうでないバッチでは単に `num_tokens` を設定します。

CUDA Graphs のラッパーはウォームアップのロジックを管理しなくなりました。ウォームアップの処理は現在、GPU の model runner が直接制御し、ウォームアップの eager 実行には `NONE` 実行時モードが割り当てられます。フル CUDA Graph 向けのウォームアップでは、ウォームアップの `dummy_run` 呼び出し中に attention を明示的に実行することも重要です。

## attention バックエンドの CUDA Graphs 互換性 { #cuda-graphs-compatibility-of-attention-backends }

attention バックエンドの CUDA Graphs 互換性を示すために、新しい列挙型 [`AttentionCGSupport`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/attention/backend/#vllm.v1.attention.backend.AttentionCGSupport) を導入しました。これは attention バックエンドの CUDA Graphs サポート能力を表す列挙型です。値は能力の順、すなわち `ALWAYS` > `UNIFORM_BATCH` > `UNIFORM_SINGLE_TOKEN_DECODE` > `NEVER` の順に並んでいます。

```python
class AttentionCGSupport(enum.Enum):
    """ Constants for the CUDA Graphs support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never CUDA Graphs supported."""

    ALWAYS = 3
    """CUDA Graphs always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """CUDA Graphs supported for batches that only contain query lengths that are
    the same, this can be used for spec-decode 
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """CUDA Graphs supported for batches that only contain query_len==1 decodes"""
    NEVER = 0
    """NO CUDA Graphs support"""
```

ハイブリッドな attention バックエンド（mamba mixer 系のモデルなど）を持つ場合を考えます。この場合、モデルの最終的な能力はすべてのバックエンドの能力の最小値で決まり、互換性のない CUDA Graphs モードは最も適したモードへダウングレードして解決されることがあります。たとえば、最小能力が `UNIFORM_BATCH` であれば `FULL` モードを `FULL_AND_PIECEWISE` モードへ、-O3 のコンパイルモードで最小能力が `NEVER` であれば `PIECEWISE` モードへダウングレードします。フォールバックの完全な方針については、[こちら](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/gpu_model_runner/#vllm.v1.worker.gpu_model_runner.GPUModelRunner._check_and_update_cudagraph_mode)のコードを参照してください。

執筆時点でフル CUDA Graphs をサポートするバックエンドを次の表に示します。

| attention バックエンド | cudagraph_support | 備考 |
| :---------------- | :---------------- | :------- |
| FlashAttention v2 | `UNIFORM_BATCH` | 実際には `ALWAYS` ですが、性能上の理由から `FULL_AND_PIECEWISE` にフォールバックする回避策をとっています |
| FlashAttention v3 | `ALWAYS` | どちらのバッチにも統一されたルーチンを持つため、`FULL` モードが適しています |
| Triton Attention | `ALWAYS` | プレフィル / 混在バッチと純粋なデコードバッチで異なるカーネルを持つため、`FULL_AND_PIECEWISE` が望ましいです |
| AITER FlashAttention | `UNIFORM_BATCH` | |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` | Blackwell 上で TRTLLM attention を使う場合は `UNIFORM_BATCH` になります |
| FlashMLA | `UNIFORM_BATCH` | |
| FlashInferMLA | `UNIFORM_BATCH` | |
| FlashInferMLASparse | `UNIFORM_BATCH` | |
| AITER MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` | |
| Mamba attention | `UNIFORM_SINGLE_TOKEN_DECODE` | |

一覧にないバックエンドはすべて `NEVER` と宣言されています。

## 使い方ガイド { #usage-guide }

現在、CLI では compilation_config の cudagraph_mode を大文字の文字列でそのまま指定します: `--compilation-config '{"cudagraph_mode": "..."}'`。`...` には `NONE`、`PIECEWISE`、`FULL`、`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE` のいずれかを指定します。`PIECEWISE` 関連のモードはすべて区分的コンパイルを必要とし、`FULL` 関連のモードはすべて attention バックエンドの CUDA Graphs サポートを必要とする点に注意してください。例:

```bash
vllm serve --model meta-llama/Llama-3.1-8B-Instruct --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

### Python の例 { #python-examples }

```python
import os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

import vllm
from vllm.config import CUDAGraphMode

compilation_config = {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}
model = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",
    compilation_config=compilation_config,
)
sampling_params = vllm.SamplingParams(
    temperature=0,  # greedy decoding
    max_tokens=1024,
)
outputs = model.generate(
    ["My name is John and"],
    sampling_params=sampling_params,
)
```

### 区分的コンパイルとグラフ全体を対象とするカスタムパス（attention 融合、シーケンス並列） { #piecewise-compilation-and-full-graph-custom-passes-attention-fusion-sequence-parallelism }

残念ながら、一部のカスタムコンパイルパスは効果を発揮するためにグラフ全体を参照する必要があり、区分的コンパイルとは両立しません。これには `AttnQuantFusionPass` と `SequenceParallelismPass` が含まれます。短期的な解決策として、attention 融合が有効な場合は（`splitting_ops=[]` を設定することで）区分的コンパイルを自動的に無効化しています。この場合、CUDA Graph のモードには（バックエンドのサポート状況に応じて）`FULL` または `FULL_DECODE_ONLY` を使います。ただし、これは別の最適化との非互換や、分かりにくい性能上のトレードオフを生みます。

長期的には、Dynamo の直後ではなく Inductor 内でグラフを分割する機能を追加しました。`CompilationConfig.use_inductor_graph_partition=True` で有効にできますが、現時点では実験的で `torch>=2.9` でのみ利用できます。この方式はグラフ全体をコンパイルする必要があり、区分的コンパイルの成果物を再利用できないため、コンパイル時間も増加します。vLLM が 2.9 をサポートしたら、区分的 CUDA graph のキャプチャも高速化されるため、これを既定の方式にする予定です。

## 性能について { #about-the-performance }

例については次のリンクを参照してください。

* [20059#issuecomment-3160858458](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3160858458)
* [20059#issuecomment-3188735226](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3188735226)
* [20059#issuecomment-3219888738](https://github.com/vllm-project/vllm/pull/20059#issuecomment-3219888738)
