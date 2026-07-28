# dstack { #dstack }

<p align="center">
    <img src="https://i.ibb.co/71kx6hW/vllm-dstack.png" alt="vLLM_plus_dstack"/>
</p>

vLLM は、任意のクラウドで LLM を動かすためのオープンソースフレームワーク [dstack](https://dstack.ai/) を使って、クラウドの GPU マシン上で実行できます。このチュートリアルでは、クラウド環境の資格情報・ゲートウェイ・GPU クォータをすでに設定済みであることを前提とします。

dstack のクライアントをインストールするには次を実行します。

```bash
pip install dstack[all]
dstack server
```

次に、dstack のプロジェクトを設定します。

```bash
mkdir -p vllm-dstack
cd vllm-dstack
dstack init
```

次に、任意の LLM（この例では `NousResearch/Llama-2-7b-chat-hf`）を載せた VM インスタンスを用意するため、dstack の `Service` 用に次の `serve.dstack.yml` を作成します。

??? code "設定"

    ```yaml
    type: service

    python: "3.11"
    env:
        - MODEL=NousResearch/Llama-2-7b-chat-hf
    port: 8000
    resources:
        gpu: 24GB
    commands:
        - pip install vllm
        - vllm serve $MODEL --port 8000
    model:
        format: openai
        type: chat
        name: NousResearch/Llama-2-7b-chat-hf
    ```

そのうえで、次の CLI を実行してプロビジョニングします。

??? console "コマンド"

    ```console
    $ dstack run . -f serve.dstack.yml

    ⠸ Getting run plan...
    Configuration  serve.dstack.yml
    Project        deep-diver-main
    User           deep-diver
    Min resources  2..xCPU, 8GB.., 1xGPU (24GB)
    Max price      -
    Max duration   -
    Spot policy    auto
    Retry policy   no

    #  BACKEND  REGION       INSTANCE       RESOURCES                               SPOT  PRICE
    1  gcp   us-central1  g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
    2  gcp   us-east1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
    3  gcp   us-west1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
        ...
    Shown 3 of 193 offers, $5.876 max

    Continue? [y/n]: y
    ⠙ Submitting run...
    ⠏ Launching spicy-treefrog-1 (pulling)
    spicy-treefrog-1 provisioning completed (running)
    Service is published at ...
    ```

プロビジョニングが終わったら、OpenAI の SDK からモデルとやり取りできます。

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="https://gateway.<gateway domain>",
        api_key="<YOUR-DSTACK-SERVER-ACCESS-TOKEN>",
    )

    completion = client.chat.completions.create(
        model="NousResearch/Llama-2-7b-chat-hf",
        messages=[
            {
                "role": "user",
                "content": "Compose a poem that explains the concept of recursion in programming.",
            }
        ],
    )

    print(completion.choices[0].message.content)
    ```

!!! note
    dstack はゲートウェイでの認証を dstack のトークンで自動的に処理します。ゲートウェイを設定したくない場合は、`Service` の代わりに dstack の `Task` をプロビジョニングできます。`Task` は開発用途のみです。dstack で vLLM をサービングする実践的な資料は[このリポジトリ](https://github.com/dstackai/dstack-examples/tree/main/deployment/vllm)（英語）を参照してください。
