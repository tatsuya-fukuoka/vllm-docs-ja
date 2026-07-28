# クイックスタート { #quickstart }

このガイドでは、vLLM を使って次のことをすばやく始める方法を説明します。

- [オフラインバッチ推論](#offline-batched-inference)
- [オンラインサービング](#online-serving)

## 前提条件 { #prerequisites }

- OS: Linux
- Python: 3.10 -- 3.13

!!! note
    vLLM は macOS でも動作します。Apple Silicon GPU による高速化には [vLLM-Metal](https://github.com/vllm-project/vllm-metal) を使用します。[GPU インストールガイド](installation/gpu.md)の「Apple Silicon」タブを参照してください。

## インストール { #installation }

=== "NVIDIA CUDA"

    NVIDIA GPU を使用している場合は、[pip](https://pypi.org/project/vllm/) で直接 vLLM をインストールできます。

    Python 環境の作成・管理には、非常に高速な環境マネージャーである [uv](https://docs.astral.sh/uv/) の使用をおすすめします。`uv` のインストール方法は[ドキュメント](https://docs.astral.sh/uv/#getting-started)を参照してください。`uv` をインストールしたら、次のコマンドで新しい Python 環境を作成し、vLLM をインストールできます。

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```

    `uv` は `--torch-backend=auto`（または `UV_TORCH_BACKEND=auto`）を指定すると、インストール済みの CUDA ドライバのバージョンを調べて[実行時に適切な PyTorch のインデックスを自動選択](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)します。特定のバックエンド（例: `cu126`）を選びたい場合は `--torch-backend=cu126`（または `UV_TORCH_BACKEND=cu126`）を指定してください。

    もう 1 つの便利な方法は `uv run` の `--with [dependency]` オプションです。永続的な環境を作らずに `vllm serve` などのコマンドを実行できます。

    ```bash
    uv run --with vllm vllm --help
    ```

    [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) で Python 環境を作成・管理することもできます。環境内で `uv` を管理したい場合は、`pip` で conda 環境に `uv` をインストールしてください。

    ```bash
    conda create -n myenv python=3.12 -y
    conda activate myenv
    pip install --upgrade uv
    uv pip install vllm --torch-backend=auto
    ```

=== "AMD ROCm"

    AMD GPU を使用している場合は、`uv` で vLLM をインストールできます。

    [uv](https://docs.astral.sh/uv/) の使用をおすすめします。追加のインデックスを[既定のインデックスより優先](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)してくれるためです。`uv` は非常に高速な Python 環境マネージャーでもあります。`uv` のインストール方法は[ドキュメント](https://docs.astral.sh/uv/#getting-started)を参照してください。`uv` をインストールしたら、次のコマンドで新しい Python 環境を作成し、vLLM をインストールできます。

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
    ```

    !!! note
        現在サポートされているのは Python 3.12、ROCm 7.0、`glibc >= 2.35` です。

    !!! note
        以前は AMD の Docker リリースパイプラインで公開された `rocm/vllm-dev` イメージが使われていましたが、vLLM の Docker リリースパイプラインに移行するため非推奨になっています。

    !!! tip
        最新の開発版をテストするための nightly Docker イメージ [vllm/vllm-openai-rocm:nightly](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags) も提供されています。

=== "Google TPU"

    Google TPU 上で vLLM を実行するには、`vllm-tpu` パッケージをインストールします。
    
    ```bash
    uv pip install vllm-tpu
    ```

    !!! note
        Docker の利用、ソースからのインストール、トラブルシューティングなど、より詳しい手順は [vLLM on TPU のドキュメント](https://docs.vllm.ai/projects/tpu/en/latest/)（英語）を参照してください。

=== "Ascend NPU"

    Ascend NPU を使用している場合は、コミュニティが保守するハードウェアプラグイン [vLLM Ascend](https://github.com/vllm-project/vllm-ascend) を通じて vLLM を実行できます。

    [vLLM Ascend のクイックスタート](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)（英語）に記載のインストール手順に従ってください。

    !!! note
        Ascend のセットアップ内容は NPU のハードウェアと CANN のバージョンによって異なります。対応バージョン、Docker イメージ、トラブルシューティングについては [vLLM Ascend のドキュメント](https://docs.vllm.ai/projects/ascend/en/latest/)（英語）を参照してください。

=== "Apple Silicon (Mac)"

    Apple Silicon の Mac を使用している場合は、Apple の Metal フレームワークを介した GPU 高速化推論のために vLLM-Metal を利用できます。

    [vLLM-Metal のドキュメント](https://github.com/vllm-project/vllm-metal#installation)に記載のインストール手順に従ってください。

    !!! note
        vLLM-Metal は計算バックエンドとして PyTorch ではなく MLX を使用するため、Hugging Face の [mlx-community](https://huggingface.co/mlx-community) にある MLX 向けに最適化されたモデルが必要です。

    !!! tip
        より詳しい手順は [GPU インストールガイド](installation/gpu.md)の「Apple Silicon」タブを参照してください。

!!! note
    CUDA 以外のプラットフォームやより詳しい情報については、[インストールガイド](installation/README.md)を参照してください。

## オフラインバッチ推論 { #offline-batched-inference }

vLLM をインストールすると、入力プロンプトのリストに対してテキスト生成を実行できます（オフラインバッチ推論）。サンプルスクリプト: [examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

この例の 1 行目では、[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスと [`SamplingParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.SamplingParams) クラスをインポートしています。

- [`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) は、vLLM エンジンでオフライン推論を実行するための主要なクラスです。
- [`SamplingParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.SamplingParams) は、サンプリング処理のパラメータを指定します。

```python
from vllm import LLM, SamplingParams
```

次のセクションでは、入力プロンプトのリストと、テキスト生成のためのサンプリングパラメータを定義します。[サンプリングの temperature](https://arxiv.org/html/2402.05201v1) は `0.8`、[nucleus sampling の確率](https://en.wikipedia.org/wiki/Top-p_sampling)は `0.95` に設定しています。サンプリングパラメータの詳細は[こちら](https://docs.vllm.ai/en/v0.26.0/api/#inference-parameters)（英語）を参照してください。

!!! important
    vLLM は既定で、Hugging Face のモデルリポジトリに `generation_config.json` があればそれを適用し、モデル作成者が推奨するサンプリングパラメータを使用します。多くの場合、[`SamplingParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.SamplingParams) を指定しなくてもこれが最良の結果をもたらします。

    vLLM 既定のサンプリングパラメータを使いたい場合は、[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) インスタンスの作成時に `generation_config="vllm"` を指定してください。

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスは、vLLM のエンジンと [OPT-125M モデル](https://arxiv.org/abs/2205.01068)をオフライン推論用に初期化します。対応モデルの一覧は[こちら](../models/supported_models.md)を参照してください。

```python
llm = LLM(model="facebook/opt-125m")
```

!!! note
    vLLM は既定で [Hugging Face](https://huggingface.co/) からモデルをダウンロードします。[ModelScope](https://www.modelscope.cn) のモデルを使いたい場合は、エンジンを初期化する前に環境変数 `VLLM_USE_MODELSCOPE` を設定してください。

    ```shell
    export VLLM_USE_MODELSCOPE=True
    ```

ここからが本題です。出力は `llm.generate` で生成します。このメソッドは入力プロンプトを vLLM エンジンの待機キューに追加し、エンジンを実行して高スループットで出力を生成します。出力は `RequestOutput` オブジェクトのリストとして返され、生成されたトークンがすべて含まれます。

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! note
    `llm.generate` メソッドは、入力プロンプトにモデルのチャットテンプレートを自動では適用しません。そのため、Instruct モデルや Chat モデルを使う場合は、期待どおりに動作させるために対応するチャットテンプレートを手動で適用する必要があります。あるいは、OpenAI の `client.chat.completions` に渡すのと同じ形式のメッセージのリストを `llm.chat` メソッドに渡すこともできます。

    ??? code
    
        ```python
        # Using tokenizer to apply chat template
        from transformers import AutoTokenizer
    
        tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate outputs
        outputs = llm.generate(texts, sampling_params)
        
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
        # Using chat interface.
        outputs = llm.chat(messages_list, sampling_params)
        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        ```

## オンラインサービング { #online-serving }

vLLM は、OpenAI API プロトコルを実装したサーバーとしてデプロイできます。これにより、OpenAI API を使っているアプリケーションの差し替え先として vLLM を利用できます。
既定では `http://localhost:8000` でサーバーを起動します。アドレスは `--host` と `--port` 引数で指定できます。サーバーは現時点では一度に 1 つのモデルをホストし、[モデル一覧](https://platform.openai.com/docs/api-reference/models/list)、[chat completion の作成](https://platform.openai.com/docs/api-reference/chat/completions/create)、[completion の作成](https://platform.openai.com/docs/api-reference/completions/create)などのエンドポイントを実装しています。

次のコマンドを実行すると、[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) モデルで vLLM サーバーを起動できます。

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

!!! note
    既定では、サーバーはトークナイザーに保存されている定義済みのチャットテンプレートを使用します。
    上書きする方法は[こちら](../serving/online_serving/README.md#chat-template)を参照してください。
!!! important
    既定では、Hugging Face のモデルリポジトリに `generation_config.json` があればサーバーがそれを適用します。つまり、一部のサンプリングパラメータの既定値がモデル作成者の推奨値で上書きされます。

    この動作を無効にするには、サーバー起動時に `--generation-config vllm` を指定してください。

このサーバーには OpenAI API と同じ形式でリクエストできます。たとえばモデルの一覧を取得するには次のようにします。

```bash
curl http://localhost:8000/v1/models
```

`--api-key` 引数または環境変数 `VLLM_API_KEY` を指定すると、サーバーがヘッダー内の API キーを検証するようになります。
`--api-key` の後には複数のキーを指定でき、サーバーはそのいずれかを受け付けます。キーのローテーションに便利です。

### vLLM で OpenAI Completions API を使う { #openai-completions-api-with-vllm }

サーバーを起動したら、入力プロンプトでモデルにリクエストできます。

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

このサーバーは OpenAI API と互換性があるため、OpenAI API を使うあらゆるアプリケーションの差し替え先として利用できます。たとえば、`openai` の Python パッケージ経由でリクエストすることもできます。

??? code

    ```python
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="San Francisco is a",
    )
    print("Completion result:", completion)
    ```

より詳しいクライアントの例はこちらにあります: [examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

### vLLM で OpenAI Chat Completions API を使う { #openai-chat-completions-api-with-vllm }

vLLM は OpenAI Chat Completions API もサポートしています。チャットインターフェイスはモデルとやり取りするためのより動的で対話的な方法であり、チャット履歴として保存できる往復のやり取りが可能です。文脈が必要なタスクや、より詳しい説明が必要なタスクに便利です。

[chat completion の作成](https://platform.openai.com/docs/api-reference/chat/completions/create)エンドポイントでモデルとやり取りできます。

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

`openai` の Python パッケージを使うこともできます。

??? code

    ```python
    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    )
    print("Chat response:", chat_response)
    ```

## Attention バックエンドについて { #on-attention-backends }

現在の vLLM は、さまざまなプラットフォームやアクセラレータのアーキテクチャで効率的に Attention を計算するための複数のバックエンドをサポートしています。システムとモデルの条件に適合する、もっとも性能の高いバックエンドが自動的に選択されます。

必要に応じて、`--attention-backend` CLI 引数で任意のバックエンドを手動で指定することもできます。

```bash
# For online serving
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

# For offline inference
python script.py --attention-backend FLASHINFER
```

指定できるバックエンドの例:

- NVIDIA CUDA: `FLASH_ATTN` または `FLASHINFER`
- AMD ROCm: `TRITON_ATTN`、`ROCM_ATTN`、`ROCM_AITER_FA`、`ROCM_AITER_UNIFIED_ATTN`、`TRITON_MLA`、`ROCM_AITER_MLA`、`ROCM_AITER_TRITON_MLA`

!!! warning
    Flash Infer を含む vLLM のビルド済み wheel は提供されていないため、あらかじめ自分の環境にインストールする必要があります。インストール方法は [Flash Infer の公式ドキュメント](https://docs.flashinfer.ai/)（英語）または [docker/Dockerfile](../../docker/Dockerfile) を参照してください。
