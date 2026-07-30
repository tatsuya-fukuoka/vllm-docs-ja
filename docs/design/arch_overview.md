# アーキテクチャ概要 { #architecture-overview }

このドキュメントでは、vLLM のアーキテクチャの概要を説明します。

[TOC]

## エントリポイント { #entrypoints }

vLLM は、システムとやり取りするためのエントリポイントをいくつか提供しています。次の図はそれらの関係を示しています。

![Entrypoints Diagram](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/arch_overview/entrypoints.excalidraw.png)

### LLM クラス { #llm-class }

LLM クラスは、オフライン推論、つまり独立したモデル推論サーバーを使わずにモデルとやり取りするための主要な Python インターフェースを提供します。

`LLM` クラスの使用例は次のとおりです。

??? code

    ```python
    from vllm import LLM, SamplingParams

    # Define a list of input prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Initialize the LLM engine with the OPT-125M model
    llm = LLM(model="facebook/opt-125m")

    # Generate outputs for the input prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the generated outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

API の詳細は、API ドキュメントの [Offline Inference](https://docs.vllm.ai/en/v0.26.0/api/#offline-inference) のセクションを参照してください。

`LLM` クラスのコードは [vllm/entrypoints/llm.py](../../vllm/entrypoints/llm.py) にあります。

### オンラインサービング { #online-serving }

vLLM のもう 1 つの主要なインターフェースは、オンラインサーバーです。このサーバーは `vllm serve` コマンドで起動できます。

```bash
vllm serve <model>
```

`vllm` の CLI のコードは [vllm/entrypoints/cli/main.py](../../vllm/entrypoints/cli/main.py) にあります。

`vllm` の CLI コマンドを介さず、API サーバーのエントリポイントが直接使われているのを見かけることもあります。例:

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

!!! warning

    `python -m vllm.entrypoints.openai.api_server` は非推奨であり、将来のリリースでサポートされなくなる可能性があります。

このコードは [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py) にあります。

API サーバーの詳細は[オンラインサービング](../serving/online_serving/README.md)のドキュメントを参照してください。

## V1 のプロセス構成 { #v1-process-architecture }

vLLM V1 は、関心事を分離しスループットを最大化するためにマルチプロセス構成を採用しています。この構成を理解しておくことは、デプロイで CPU リソースを適切にサイジングするうえで重要です。主なプロセスは次のとおりです。

### API サーバープロセス { #api-server-process }

API サーバープロセスは HTTP リクエスト（OpenAI 互換 API など）を処理し、入力の前処理（トークン化、マルチモーダルデータの読み込み）を行い、結果をクライアントへストリーミングします。エンジンコアのプロセスとは ZMQ ソケットを通じて通信します。

既定では **API サーバープロセスは 1 つ**ですが、データ並列を使う場合は API サーバー数がデータ並列サイズに合わせて自動的にスケールします。`--api-server-count` フラグで手動設定することもできます。各 API サーバーは多対多のトポロジで ZMQ を介して**すべての**エンジンコアに接続するため、どの API サーバーからでもどのエンジンコアへリクエストをルーティングできます。各 API サーバープロセスは、メディアの読み込みに複数の CPU スレッドを使います（`VLLM_MEDIA_LOADING_THREAD_COUNT` で制御、既定 8）。

コードは [vllm/entrypoints/openai/api_server.py](../../vllm/entrypoints/openai/api_server.py) と [vllm/v1/utils.py](../../vllm/v1/utils.py) にあります。

### エンジンコアプロセス { #engine-core-process }

エンジンコアプロセスはスケジューラを実行し、KV キャッシュを管理し、GPU ワーカー間のモデル実行を調整します。リクエストを継続的にスケジュールし、GPU ワーカーへ処理を割り当てるビジーループを回します。

**データ並列のランクごとにエンジンコアプロセスが 1 つ**あります。たとえば `--data-parallel-size 4` の場合、エンジンコアプロセスは 4 つになります。

コードは [vllm/v1/engine/core.py](../../vllm/v1/engine/core.py) と [vllm/v1/engine/utils.py](../../vllm/v1/engine/utils.py) にあります。

### GPU ワーカープロセス { #gpu-worker-processes }

各 GPU は専用のワーカープロセスによって管理されます。ワーカープロセスはモデルの重みを読み込み、forward パスを実行し、GPU メモリを管理します。ワーカーは、自分を所有するエンジンコアのプロセスと通信します。

**GPU ごとにワーカープロセスが 1 つ**あります。GPU ワーカープロセスの総数は、エンジンコアごとに `tensor_parallel_size × pipeline_parallel_size` となります。

コードは [vllm/v1/executor/multiproc_executor.py](../../vllm/v1/executor/multiproc_executor.py) と [vllm/v1/worker/gpu_worker.py](../../vllm/v1/worker/gpu_worker.py) にあります。

### DP コーディネータープロセス（条件付き） { #dp-coordinator-process-conditional }

データ並列（`--data-parallel-size > 1`）を使う場合、DP ランク間の負荷分散を管理し、MoE モデルの forward パスの同期を調整する追加のコーディネータープロセスが起動します。

**DP コーディネータープロセスは 1 つ**です（データ並列が有効な場合のみ）。

コードは [vllm/v1/engine/coordinator.py](../../vllm/v1/engine/coordinator.py) にあります。

### プロセス数のまとめ { #process-count-summary }

GPU 数が `N`、テンソル並列サイズが `TP`、データ並列サイズが `DP`、API サーバー数が `A` のデプロイの場合は次のようになります。

| プロセスの種類 | 個数 | 備考 |
| - | - | - |
| API サーバー | `A`（既定は `DP`） | HTTP リクエストと入力の前処理を担当 |
| エンジンコア | `DP`（既定は 1） | スケジューラと KV キャッシュの管理 |
| GPU ワーカー | `N`（= `DP × PP × TP`） | GPU ごとに 1 つ。モデルの forward パスを実行 |
| DP コーディネーター | `DP > 1` なら 1、そうでなければ 0 | DP ランク間の負荷分散 |
| **合計** | **`A + DP + N`（DP > 1 の場合は +1）** | |

たとえば、GPU 4 台の一般的な単一ノードのデプロイ（`vllm serve -tp=4`）では次のようになります。

- API サーバー 1 + エンジンコア 1 + GPU ワーカー 4 = **6 プロセス**

<figure markdown="1">
![V1 Process Architecture - TP=4](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/arch_overview/v1_process_architecture_tp4.png)
</figure>

GPU 8 台のデータ並列デプロイ（`vllm serve -tp=2 -dp=4`）では次のようになります。

- API サーバー 4 + エンジンコア 4 + GPU ワーカー 8 + DP コーディネーター 1 = **17 プロセス**

<figure markdown="1">
![V1 Process Architecture - TP=2, DP=4](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/arch_overview/v1_process_architecture_tp2_dp4.png)
</figure>

CPU リソースのサイジングの推奨事項は、[GPU デプロイにおける CPU リソース](../configuration/optimization.md#cpu-resources-for-gpu-deployments)を参照してください。

## LLM エンジン { #llm-engine }

`LLMEngine` と `AsyncLLMEngine` のクラスは vLLM システムの動作の中心にあり、モデルの推論と非同期のリクエスト処理を担当します。

![LLMEngine Diagram](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/arch_overview/llm_engine.excalidraw.png)

### LLMEngine { #llmengine }

`LLMEngine` クラスは vLLM エンジンの中核となるコンポーネントです。クライアントからリクエストを受け取り、モデルから出力を生成する役割を担います。`LLMEngine` には、入力の前処理、モデルの実行（複数ホストや複数 GPU にまたがる分散実行を含む）、スケジューリング、出力の後処理が含まれます。

- **入力の前処理**: 指定されたトークナイザーを使って入力テキストをトークン化します。
- **スケジューリング**: 各ステップでどのリクエストを処理するかを選びます。
- **モデルの実行**: 複数 GPU にまたがる分散実行を含め、言語モデルの実行を管理します。
- **出力の後処理**: モデルが生成した出力を処理し、言語モデルからのトークン ID を人が読めるテキストにデコードします。

`LLMEngine` のコードは [vllm/engine/llm_engine.py](../../vllm/engine/llm_engine.py) にあります。

### AsyncLLMEngine { #asyncllmengine }

`AsyncLLMEngine` クラスは `LLMEngine` クラスの非同期ラッパーです。`asyncio` を使ってバックグラウンドループを作り、届いたリクエストを継続的に処理します。`AsyncLLMEngine` はオンラインサービング向けに設計されており、複数の同時リクエストを処理してクライアントへ出力をストリーミングできます。

OpenAI 互換 API サーバーは `AsyncLLMEngine` を使います。より簡単な例としてのデモ用 API サーバーも [examples/applications/api_server/server.py](../../examples/applications/api_server/server.py) にあります。

`AsyncLLMEngine` のコードは [vllm/engine/async_llm_engine.py](../../vllm/engine/async_llm_engine.py) にあります。

## ワーカー { #worker }

ワーカーはモデル推論を実行するプロセスです。vLLM は、GPU などのアクセラレータデバイス 1 台を 1 プロセスで制御するという一般的な慣習に従っています。たとえば、テンソル並列サイズ 2 とパイプライン並列サイズ 2 を使う場合、ワーカーは合計 4 つになります。ワーカーは `rank` と `local_rank` で識別されます。`rank` は全体のオーケストレーションに使われ、`local_rank` は主にアクセラレータデバイスの割り当てや、ファイルシステムや共有メモリといったローカルリソースへのアクセスに使われます。

## モデルランナー { #model-runner }

各ワーカーはモデルランナーのオブジェクトを 1 つ持ち、モデルの読み込みと実行を担当します。入力テンソルの準備や cudagraph のキャプチャなど、モデル実行のロジックの多くはここにあります。

## モデル { #model }

各モデルランナーのオブジェクトはモデルオブジェクトを 1 つ持ちます。これが実際の `torch.nn.Module` のインスタンスです。さまざまな設定が最終的に得られるクラスにどう影響するかは、[Hugging Face との統合](huggingface_integration.md)を参照してください。

## クラス階層 { #class-hierarchy }

次の図は vLLM のクラス階層を示しています。

![Class Hierarchy](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/hierarchy.png)

このクラス階層の背後には、いくつかの重要な設計判断があります。

1\. **拡張性**: 階層内のすべてのクラスは、必要な情報をすべて含む設定オブジェクトを受け取ります。受け渡される主要な設定オブジェクトは [VllmConfig](https://github.com/vllm-project/vllm/blob/d1c6799b8870e513bf4f2305cbf6cda9fc3d773b/vllm/config.py#L2036) クラスです。クラス階層はかなり深く、各クラスは自分が必要とする設定を読み取る必要があります。すべての設定を 1 つのオブジェクトにまとめることで、設定オブジェクトを簡単に受け渡し、必要な設定にアクセスできます。たとえば、モデルランナーだけに関わる新機能を追加したいとします（LLM 推論の分野の進化の速さを考えると、これはよくあることです）。その場合、`VllmConfig` クラスに新しい設定オプションを追加する必要があります。設定オブジェクト全体を受け渡しているため、`VllmConfig` クラスに設定オプションを追加するだけで、モデルランナーから直接アクセスできます。新しい設定オプションを渡すために、エンジン・ワーカー・モデルの各クラスのコンストラクタを変更する必要はありません。

2\. **統一性**: モデルランナーには、モデルを生成・初期化するための統一されたインターフェースが必要です。vLLM は 50 種類以上の人気のあるオープンソースモデルをサポートしており、各モデルは独自の初期化ロジックを持ちます。コンストラクタのシグネチャがモデルごとに異なると、複雑で誤りやすい検査ロジックなしには、モデルランナーはコンストラクタを適切に呼び出せません。モデルクラスのコンストラクタを統一することで、モデルランナーは具体的なモデルの種類を知らなくても、簡単にモデルを生成・初期化できます。これはモデルを組み合わせる際にも有用です。vision-language モデルは、多くの場合ビジョンモデルと言語モデルから構成されます。コンストラクタを統一しておけば、ビジョンモデルと言語モデルを簡単に生成し、それらを組み合わせて vision-language モデルにできます。

!!! note
    この変更に対応するため、vLLM のすべてのモデルのシグネチャは次のように更新されました。

    ```python
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
    ```

    誤った引数を渡してしまうのを防ぐため、コンストラクタはキーワード専用になりました。これにより、古い設定が渡された場合はコンストラクタがエラーを送出します。vLLM の開発者は vLLM 内のすべてのモデルについてこの変更をすでに適用済みです。ツリー外で登録されたモデルについては、開発者が自分のモデルを更新する必要があります。たとえば、古いコンストラクタのシグネチャを新しいものに適合させるシムのコードを追加します。

    ??? code

        ```python
        class MyOldModel(nn.Module):
            def __init__(
                self,
                config,
                cache_config: Optional[CacheConfig] = None,
                quant_config: Optional[QuantizationConfig] = None,
                lora_config: Optional[LoRAConfig] = None,
                prefix: str = "",
            ) -> None:
                ...

        from vllm.config import VllmConfig
        class MyNewModel(MyOldModel):
            def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
                config = vllm_config.model_config.hf_config
                cache_config = vllm_config.cache_config
                quant_config = vllm_config.quant_config
                lora_config = vllm_config.lora_config
                super().__init__(config, cache_config, quant_config, lora_config, prefix)

        from packaging import version
        if version.parse(__version__) >= version.parse("0.6.4"):
            MyModel = MyNewModel
        else:
            MyModel = MyOldModel
        ```

    このようにすることで、モデルは vLLM の新旧両方のバージョンで動作します。

3\. **初期化時のシャーディングと量子化**: 一部の機能はモデルの重みを変更する必要があります。たとえば、テンソル並列はモデルの重みをシャーディングする必要があり、量子化はモデルの重みを量子化する必要があります。この機能の実装方法は 2 つ考えられます。1 つはモデルの初期化後に重みを変更する方法、もう 1 つはモデルの初期化中に重みを変更する方法です。vLLM は後者を選びました。前者のアプローチは大きなモデルにスケールしません。たとえば 405B のモデル（重みは約 810GB）を 16 台の H100 80GB GPU で動かしたいとします。理想的には、各 GPU は 50GB 分の重みだけを読み込むべきです。モデルの初期化後に重みを変更する場合、810GB の重み全体を各 GPU に読み込んでからシャーディングすることになり、膨大なメモリのオーバーヘッドが生じます。一方、モデルの初期化中にシャーディングすれば、各層は必要な分のシャードだけを作るため、メモリのオーバーヘッドははるかに小さくなります。量子化にも同じ考え方が当てはまります。なお、モデルが prefix に応じて異なる初期化を行えるよう、モデルのコンストラクタには追加の引数 `prefix` も設けています。これは、モデルの部分ごとに異なる量子化を行う不均一な量子化で有用です。`prefix` は通常、最上位のモデルでは空文字列で、サブモデルでは `"vision"` や `"language"` のような文字列になります。一般に、これはチェックポイントファイル内のモジュールの state dict の名前と一致します。

この設計の欠点の 1 つは、各コンポーネントが完全な設定オブジェクトで初期化される必要があるため、vLLM の個々のコンポーネントに対するユニットテストを書きにくいことです。この問題には、すべてのフィールドを `None` に設定した既定の設定オブジェクトを作る初期化関数を提供することで対処しています。テストしたいコンポーネントが設定オブジェクトのごく一部のフィールドしか参照しない場合、既定の設定オブジェクトを作り、関心のあるフィールドだけを設定すればよいのです。これにより、コンポーネントを単独でテストできます。なお、vLLM のテストの多くはシステム全体を対象とするエンドツーエンドのテストであるため、これは大きな問題にはなっていません。

まとめると、完全な設定オブジェクトである `VllmConfig` は、すべての vLLM のクラスで共有されるエンジンレベルのグローバルな状態と見なせます。
