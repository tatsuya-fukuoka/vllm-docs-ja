# Hugging Face Inference Endpoints { #hugging-face-inference-endpoints }

## 概要 { #overview }

vLLM に対応したモデルは、[Hugging Face Hub](https://huggingface.co) から、あるいは [Inference Endpoints](https://endpoints.huggingface.co) から直接、Hugging Face Inference Endpoints にデプロイできます。

vLLM の統合とデプロイオプションの詳細は[高度なデプロイの詳細](#advanced-deployment-details)を参照してください。

## デプロイの方法 { #deployment-methods }

- [**Method 1: Deploy from the Catalog.**](#method-1-deploy-from-the-catalog) One-click deploy models from the Hugging Face Hub with ready-made optimized configurations.
- [**Method 2: Guided Deployment (Transformers Models).**](#method-2-guided-deployment-transformers-models) Instantly deploy models tagged with `transformers` from the Hub UI using the **Deploy** button.
- [**Method 3: Manual Deployment (Advanced Models).**](#method-3-manual-deployment-advanced-models) For models that either use custom code with the `transformers` tag, or don’t run with standard `transformers` but are supported by vLLM. This method requires manual configuration.

### 方法 1: カタログからデプロイする { #method-1-deploy-from-the-catalog }

この方法は、Hugging Face Inference Endpoints で vLLM を使い始めるもっとも簡単な方法です。検証済みかつ最適化されたデプロイ設定を持つモデルのカタログを [Inference Endpoints](https://endpoints.huggingface.co/catalog) で閲覧できます。

1. [Endpoints Catalog](https://endpoints.huggingface.co/catalog) を開き、**Inference Server** のオプションで `vLLM` を選びます。最適化済みの設定が用意されたモデルの一覧が表示されます。

    ![Endpoints Catalog](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-catalog.png)

1. 使いたいモデルを選び、**Create Endpoint** をクリックします。

    ![Create Endpoint](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-create-endpoint.png)

1. デプロイが完了するとエンドポイントを利用できます。コンソールに表示された URL で `DEPLOYMENT_URL` を更新してください。必要に応じて末尾に `/v1` を付けます。

    ```python
    # pip install openai
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=DEPLOYMENT_URL,
        api_key=os.environ["HF_TOKEN"],  # https://huggingface.co/settings/tokens
    )

    chat_completion = client.chat.completions.create(
        model="HuggingFaceTB/SmolLM3-3B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Give me a brief explanation of gravity in simple terms.",
                    }
                ],
            }
        ],
        stream=True,
    )

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    ```

!!! note
    カタログには、GPU の設定や推論エンジンの構成を含め、vLLM 向けに最適化されたモデルが並んでいます。エンドポイントの監視や、**コンテナとその設定**の更新は Inference Endpoints の UI から行えます。

### 方法 2: ガイド付きデプロイ（Transformers のモデル） { #method-2-guided-deployment-transformers-models }

この方法は、メタデータに [`transformers` ライブラリのタグ](https://huggingface.co/models?library=transformers)が付いたモデルに適用できます。手動の設定なしに、Hub の UI から直接モデルをデプロイできます。

1. [Hugging Face Hub](https://huggingface.co/models) でモデルのページを開きます。  
   この例では [`ibm-granite/granite-docling-258M`](https://huggingface.co/ibm-granite/granite-docling-258M) モデルを使います。対応しているかどうかは、[README](https://huggingface.co/ibm-granite/granite-docling-258M/blob/main/README.md) の front matter で確認できます。

2. **Deploy** ボタンを探します。`transformers` タグが付いたモデルでは、[モデルカード](https://huggingface.co/ibm-granite/granite-docling-258M)の右上に表示されます。

    ![Locate deploy button](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-locate-deploy-button.png)

3. **Deploy** ボタン > **HF Inference Endpoints** をクリックします。Inference Endpoints の画面に移動し、デプロイを設定できます。

    ![Click deploy button](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-click-deploy-button.png)

4. ハードウェア（例では AWS > GPU > T4）とコンテナの設定を選びます。コンテナの種類として `vLLM` を選び、**Create Endpoint** を押してデプロイを確定します。

    ![Select Hardware](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-select-hardware.png)

5. デプロイしたエンドポイントを使います。コンソールに表示された URL で `DEPLOYMENT_URL` を更新してください（必要に応じて `/v1` を追加）。プログラムからでも SDK 経由でも利用できます。

    ```python
    # pip install openai
    from openai import OpenAI
    import os

    client = OpenAI(
        base_url=DEPLOYMENT_URL,
        api_key=os.environ["HF_TOKEN"],  # https://huggingface.co/settings/tokens
    )

    chat_completion = client.chat.completions.create(
        model="ibm-granite/granite-docling-258M",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://huggingface.co/ibm-granite/granite-docling-258M/resolve/main/assets/new_arxiv.png",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Convert this page to docling.",
                    },
                ]
            }
        ],
        stream=True,
    )

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")
    ```

!!! note
    この方法では推定された既定値が使われます。要件に合わせて設定の調整が必要になる場合があります。

### 方法 3: 手動デプロイ（応用的なモデル） { #method-3-manual-deployment-advanced-models }

次の理由から、手動でのデプロイが必要なモデルがあります。

- `transformers` タグが付いているが独自コードを使っている
- 標準の `transformers` では動かないが `vLLM` はサポートしている

これらのモデルは、モデルカードの **Deploy** ボタンからはデプロイできません。

このガイドでは、vLLM に統合された OCR モデル [`rednote-hilab/dots.ocr`](https://huggingface.co/rednote-hilab/dots.ocr) を例に、手動でのデプロイを説明します（vLLM の [PR](https://github.com/vllm-project/vllm/pull/24645) を参照）。

1. 新しいデプロイを開始します。[Inference Endpoints](https://endpoints.huggingface.co/) を開き `New` をクリックします。

    ![New Endpoint](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-new-endpoint.png)

2. Hub でモデルを検索します。ダイアログで **Hub** に切り替え、目的のモデルを検索します。

    ![Select model](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-select-model.png)

3. インフラを選びます。設定ページで、利用可能な選択肢からクラウドプロバイダとハードウェアを選びます。  
   このデモでは AWS と L4 GPU を選びます。必要なハードウェアに応じて調整してください。

    ![Choose Infra](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-choose-infra.png)

4. コンテナを設定します。**Container Configuration** までスクロールし、コンテナの種類として `vLLM` を選びます。

    ![Configure Container](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/hf-inference-endpoints-configure-container.png)

5. エンドポイントを作成します。**Create Endpoint** をクリックしてモデルをデプロイします。

    エンドポイントの準備ができたら、OpenAI Completion API、cURL、その他の SDK から利用できます。必要に応じてデプロイ URL の末尾に `/v1` を付けてください。

!!! note
    **コンテナの設定**（Container URI、Container Arguments）は Inference Endpoints の UI から変更し、**Update Endpoint** を押して反映できます。これにより、更新後のコンテナ設定でエンドポイントが再デプロイされます。

## 高度なデプロイの詳細 { #advanced-deployment-details }

[Transformers モデリングバックエンドの統合](https://blog.vllm.ai/2025/04/11/transformers-backend.html)により、vLLM は `transformers` に対応したあらゆるモデルを公開初日からサポートするようになりました。つまり、そうしたモデルを Hugging Face Inference Endpoints にデプロイできます。

Hugging Face Inference Endpoints は、vLLM でモデルをサービングするためのフルマネージドな環境を提供します。サーバーの設定、依存関係のインストール、クラスタの管理なしにモデルをデプロイできます。オートスケーリングにも対応しています。

このプラットフォームは Hugging Face Hub とシームレスに統合されており、vLLM または `transformers` に対応した任意のモデルをデプロイし、利用状況を追跡し、推論エンジンを直接更新できます。vLLM エンジンはあらかじめ設定済みで、そのまま利用できます。

## 次のステップ { #next-steps }

- [Inference Endpoints](https://endpoints.huggingface.co/catalog) のモデルカタログを見る
- Inference Endpoints の[ドキュメント](https://huggingface.co/docs/inference-endpoints/en/index)を読む
- [Inference Endpoints のエンジン](https://huggingface.co/docs/inference-endpoints/en/engines/vllm)について学ぶ
- [Transformers モデリングバックエンドの統合](https://blog.vllm.ai/2025/04/11/transformers-backend.html)を理解する
