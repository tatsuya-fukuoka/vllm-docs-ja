<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

Apple Silicon 上で GPU アクセラレーション推論を行うには、[vLLM-Metal](https://github.com/vllm-project/vllm-metal) を使います。これは MLX を計算バックエンドとして使い、Apple の Metal フレームワークを通じてネイティブな GPU アクセラレーションを提供する、コミュニティ管理のハードウェアプラグインです。

vLLM-Metal は、Hugging Face の [mlx-community](https://huggingface.co/mlx-community) organization にある MLX 最適化済みモデルで動作します。ここでは、Apple Silicon 向けに最適化された人気モデルの量子化版が提供されています。

!!! tip
    インストールと使い方については、後述の [vLLM-Metal を使ったセットアップ](#set-up-using-vllm-metal)のセクションを参照してください。

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: macOS Sonoma 以降
- ハードウェア: Apple Silicon
- Metal のサポートが有効であること

!!! note
    インストール手順は、後述の [vLLM-Metal を使ったセットアップ](#set-up-using-vllm-metal)のセクションを参照してください。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

## vLLM-Metal を使ったセットアップ { #set-up-using-vllm-metal }

vLLM-Metal は、Apple Silicon 上でネイティブな GPU アクセラレーションを提供する独立したパッケージとして配布されています。

vLLM-Metal をインストールするには、[vLLM-Metal のドキュメント](https://github.com/vllm-project/vllm-metal#installation)のインストール手順に従ってください。

インストールでは次のことが行われます。

1. 適切な Python 環境をセットアップする
2. MLX と必要な依存パッケージをインストールする
3. vLLM-Metal パッケージをインストールする

インストール後は、Metal による GPU アクセラレーションを使って vLLM を利用できます。

!!! tip
    vLLM-Metal を使う場合、最良の性能を得るには Hugging Face の [mlx-community](https://huggingface.co/mlx-community) のモデルを使ってください。これらのモデルは MLX 向けに最適化されており、Apple Silicon 上で効率的に動作する量子化版（4 ビット、8 ビット）が用意されていることも多いです。

    モデルの例: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

### vLLM-Metal の使い方 { #using-vllm-metal }

インストール後、vLLM-Metal は OpenAI 互換 API サーバーを起動するための使いやすい CLI を提供します。

```bash
# Activate the vLLM-Metal environment
source ~/.venv-vllm-metal/bin/activate

# Start the API server (specify your mlx-community model or it will use default)
vllm serve
```

サーバーが起動したら、いくつかの方法でやり取りできます。

#### 方法 1: 対話的なチャット { #option-1-interactive-chat }

新しいターミナルを開き、対話的なチャットセッションを開始します。

```bash
source ~/.venv-vllm-metal/bin/activate
vllm chat
```

#### 方法 2: curl による API リクエスト { #option-2-api-requests-with-curl }

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

#### 方法 3: OpenAI SDK を使った Python { #option-3-python-with-openai-sdk }

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # No auth required for local server
)

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

`vllm` の CLI コマンドの詳細は、[OpenAI 互換サーバーのドキュメント](../../serving/online_serving/openai_compatible_server.md)を参照してください。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

vLLM-Metal は vLLM-Metal パッケージとしてインストールします。上記の [vLLM-Metal を使ったセットアップ](#set-up-using-vllm-metal)のセクションを参照してください。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

ソースからのビルド手順は、[vLLM-Metal のドキュメント](https://github.com/vllm-project/vllm-metal#installation)を参照してください。

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

vLLM-Metal は次を提供します。

- Metal を使ったネイティブな GPU アクセラレーション
- Apple Silicon 向けに最適化された MLX ベースの計算バックエンド
- OpenAI 互換の API サーバー
- 主要なモデルアーキテクチャのサポート

個別の機能サポートと制限については、[vLLM-Metal のドキュメント](https://github.com/vllm-project/vllm-metal)を参照してください。

--8<-- [end:supported-features]
