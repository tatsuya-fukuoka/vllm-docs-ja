# Cerebrium { #cerebrium }

<p align="center">
    <img src="https://i.ibb.co/hHcScTT/Screenshot-2024-06-13-at-10-14-54.png" alt="vLLM_plus_cerebrium"/>
</p>

vLLM は、AI アプリケーションの構築とデプロイを容易にするサーバーレスの AI インフラプラットフォーム [Cerebrium](https://www.cerebrium.ai/) を使って、クラウドの GPU マシン上で実行できます。

Cerebrium のクライアントをインストールするには次を実行します。

```bash
pip install cerebrium
cerebrium login
```

次に、Cerebrium のプロジェクトを作成します。

```bash
cerebrium init vllm-project
```

次に、必要なパッケージをインストールするため、cerebrium.toml に以下を追加します。

```toml
[cerebrium.deployment]
docker_base_image_url = "nvidia/cuda:12.1.1-runtime-ubuntu22.04"

[cerebrium.dependencies.pip]
vllm = "latest"
```

続いて、任意の LLM（この例では `mistralai/Mistral-7B-Instruct-v0.1`）の推論を行うコードを `main.py` に追加します。

??? code

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

    def run(prompts: list[str], temperature: float = 0.8, top_p: float = 0.95):

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
        outputs = llm.generate(prompts, sampling_params)

        # Print the outputs.
        results = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            results.append({"prompt": prompt, "generated_text": generated_text})

        return {"results": results}
    ```

そのうえで、次のコマンドを実行してクラウドにデプロイします。

```bash
cerebrium deploy
```

成功すると、推論を呼び出すための curl コマンドが返されます。URL の末尾には呼び出す関数名（この例では `/run`）を付ける点に注意してください。

??? console "コマンド"

    ```bash
    curl -X POST https://api.cortex.cerebrium.ai/v4/p-xxxxxx/vllm/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: <JWT TOKEN>' \
    --data '{
    "prompts": [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is"
    ]
    }'
    ```

次のようなレスポンスが返ります。

??? console "レスポンス"

    ```json
    {
        "run_id": "52911756-3066-9ae8-bcc9-d9129d1bd262",
        "result": {
            "result": [
                {
                    "prompt": "Hello, my name is",
                    "generated_text": " Sarah, and I'm a teacher. I teach elementary school students. One of"
                },
                {
                    "prompt": "The president of the United States is",
                    "generated_text": " elected every four years. This is a democratic system.\n\n5. What"
                },
                {
                    "prompt": "The capital of France is",
                    "generated_text": " Paris.\n"
                },
                {
                    "prompt": "The future of AI is",
                    "generated_text": " bright, but it's important to approach it with a balanced and nuanced perspective."
                }
            ]
        },
        "run_time_ms": 152.53663063049316
    }
    ```

これで、使った分の計算リソースにだけ課金される、オートスケーリング対応のエンドポイントが手に入りました。
