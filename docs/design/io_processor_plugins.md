# IO プロセッサプラグイン { #io-processor-plugins }

IO プロセッサプラグインは、プーリングモデルにおいてモデルの入出力に対する前処理・後処理を行える機能です。ユーザーが独自の入力を vLLM に渡すと、それが 1 つ以上のモデルプロンプトに変換され、モデルの `encode` メソッドに渡される、という考え方です。こうしたプラグインの活用例としては、vLLM をマルチモーダルデータの生成に使うケースが挙げられます。たとえば、ユーザーが画像を vLLM に入力し、出力として画像を得る、といったものです。

IO プロセッサプラグインを使って推論を行う場合、プロンプトの型はプラグインが定義し、最終的なリクエスト出力も同様です。vLLM は入出力データの検証を一切行わないため、モデルに正しいデータが渡され、ユーザーに正しいデータが返されることを保証するのはプラグインの責任です。現時点でこれらのプラグインはプーリングモデルのみをサポートしており、`LLM` および `AsyncLLM` の `encode` メソッド、あるいはオンラインサービングモードでは `/pooling` エンドポイントから呼び出せます。

## IO プロセッサプラグインの作成 { #writing-an-io-processor-plugin }

IO プロセッサプラグインは [`IOProcessor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/plugins/io_processors/interface/#vllm.plugins.io_processors.interface.IOProcessor) インターフェースを実装します。

```python
IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")

class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """Abstract interface for pre/post-processing of engine I/O."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__()

        self.vllm_config = vllm_config

    def parse_data(self, data: object) -> IOProcessorInput:
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # We cannot guarantee outputs are returned in the same order they were
        # fed to vLLM.
        # Let's sort them by id before post_processing
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)
```

`parse_data` メソッドは、ユーザーのデータを検証し、`pre_process*` メソッドが期待する入力に変換するために使います。
`merge_sampling_params` と `merge_pooling_params` の各メソッドは、入力された `SamplingParams` または `PoolingParams`（あれば）を既定値とマージします。
`pre_process*` メソッドは、検証済みのプラグイン入力を受け取り、通常の推論用に vLLM のモデルプロンプトを生成します。
`post_process*` メソッドは、`PoolingRequestOutput` オブジェクトを入力として受け取り、プラグイン独自の出力を生成します。

PrithviGeospatialMAE モデルで geotiff 画像を生成できるようにするプラグインの実装例は[こちら](https://github.com/IBM/terratorch/tree/main/terratorch/vllm/plugins/segmentation)にあります。オンライン推論の例（[examples/pooling/plugin/prithvi_geospatial_mae_online.py](../../examples/pooling/plugin/prithvi_geospatial_mae_online.py)）とオフライン推論の例（[examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py](../../examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py)）も参照してください。

## IO プロセッサプラグインの利用 { #using-an-io-processor-plugin }

IO プロセッサプラグインはエンジンの起動時に読み込まれます。読み込むプラグイン名を指定する方法は 2 つあります。

1. vLLM の `EngineArgs` を使う: `AsyncLLM` の初期化に使う `EngineArgs` の `io_processor_plugin` 引数を設定します。オフラインモードでは `LLM` に `io_processor_plugin` 引数を渡すこと、サービングモードでは `--io-processor-plugin` 引数を渡すことでも同じことができます。
2. モデルの HF 設定を使う: モデルの設定ファイル（config.json）に `io_processor_plugin` フィールドを追加します。

この順序は優先度も表します。つまり、`EngineArgs` で指定したプラグイン名は、モデルの HF 設定（config.json）で指定されたプラグイン名より優先されます。
