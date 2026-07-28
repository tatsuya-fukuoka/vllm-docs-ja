# SkyPilot { #skypilot }

<p align="center">
  <img src="https://imgur.com/yxtzPEu.png" alt="vLLM"/>
</p>

任意のクラウドで LLM を動かすためのオープンソースフレームワーク [SkyPilot](https://github.com/skypilot-org/skypilot) を使うと、vLLM を**クラウドや Kubernetes 上で実行し、複数のサービスレプリカへスケールさせる**ことができます。Llama-3 や Mixtral などさまざまなオープンモデルの例は [SkyPilot AI gallery](https://skypilot.readthedocs.io/en/latest/gallery/index.html)（英語）にあります。

## 前提条件 { #prerequisites }

- [HuggingFace のモデルページ](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)で `meta-llama/Meta-Llama-3-8B-Instruct` へのアクセスを申請します。
- SkyPilot がインストール済みであることを確認します（[ドキュメント](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)）。
- `sky check` でクラウドまたは Kubernetes が有効になっていることを確認します。

```bash
pip install skypilot-nightly
sky check
```

## 単一インスタンスで実行する { #run-on-a-single-instance }

サービング用の vLLM SkyPilot YAML は [serving.yaml](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm/serve.yaml) を参照してください。

??? code "YAML"

    ```yaml
    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.
      use_spot: True
      disk_size: 512  # Ensure model checkpoints can fit.
      disk_tier: best
      ports: 8081  # Expose to internet traffic.

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # Change to your own huggingface token, or use --env to pass.

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # Install Gradio for web UI.
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo 'Starting vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log &

      echo 'Waiting for vllm api server to start...'
      while ! `cat api_server.log | grep -q 'Uvicorn running on'`; do sleep 1; done

      echo 'Starting gradio server...'
      git clone https://github.com/vllm-project/vllm.git || true
      python vllm/examples/applications/chatbot/gradio_openai_chatbot_webserver.py \
        -m $MODEL_NAME \
        --port 8811 \
        --model-url http://localhost:8081/v1 \
        --stop-token-ids 128009,128001
    ```

候補として挙げた GPU（L4、A10g など）のいずれかで Llama-3 8B モデルのサービングを開始します。

```bash
HF_TOKEN="your-huggingface-token" sky launch serving.yaml --env HF_TOKEN
```

コマンドの出力を確認してください。共有可能な gradio のリンク（次の出力の最終行のようなもの）が表示されます。ブラウザで開くと、LLaMA モデルでテキスト補完を試せます。

```console
(task, pid=7431) Running on public URL: https://<gradio-hash>.gradio.live
```

**任意**: 既定の 8B ではなく 70B モデルをサービングし、より多くの GPU を使う場合:

```bash
HF_TOKEN="your-huggingface-token" \
  sky launch serving.yaml \
  --gpus A100:8 \
  --env HF_TOKEN \
  --env MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
```

## 複数レプリカへのスケールアップ { #scale-up-to-multiple-replicas }

SkyPilot は、組み込みのオートスケーリング・負荷分散・耐障害性の機能により、サービスを複数のレプリカへスケールできます。YAML ファイルに services セクションを追加するだけです。

??? code "YAML"

    ```yaml
    service:
      replicas: 2
      # An actual request for readiness probe.
      readiness_probe:
        path: /v1/chat/completions
        post_data:
        model: $MODEL_NAME
        messages:
          - role: user
            content: Hello! What is your name?
      max_completion_tokens: 1
    ```

??? code "YAML"

    ```yaml
    service:
      replicas: 2
      # An actual request for readiness probe.
      readiness_probe:
        path: /v1/chat/completions
        post_data:
          model: $MODEL_NAME
          messages:
            - role: user
              content: Hello! What is your name?
          max_completion_tokens: 1

    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.
      use_spot: True
      disk_size: 512  # Ensure model checkpoints can fit.
      disk_tier: best
      ports: 8081  # Expose to internet traffic.

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # Change to your own huggingface token, or use --env to pass.

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # Install Gradio for web UI.
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo 'Starting vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log
    ```

複数レプリカで Llama-3 8B モデルのサービングを開始します。

```bash
HF_TOKEN="your-huggingface-token" \
  sky serve up -n vllm serving.yaml \
  --env HF_TOKEN
```

サービスの準備が完了するまで待ちます。

```bash
watch -n10 sky serve status vllm
```

出力例:

```console
Services
NAME  VERSION  UPTIME  STATUS  REPLICAS  ENDPOINT
vllm  1        35s     READY   2/2       xx.yy.zz.100:30001

Service Replicas
SERVICE_NAME  ID  VERSION  IP            LAUNCHED     RESOURCES                STATUS  REGION
vllm          1   1        xx.yy.zz.121  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
vllm          2   1        xx.yy.zz.245  18 mins ago  1x GCP([Spot]{'L4': 1})  READY   us-east4
```

サービスが READY になると、サービス用の単一エンドポイントが得られ、そこからアクセスできます。

??? console "コマンド"

    ```bash
    ENDPOINT=$(sky serve status --endpoint 8081 vllm)
    curl -L http://$ENDPOINT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Who are you?"
        }
        ],
        "stop_token_ids": [128009,  128001]
      }'
    ```

オートスケーリングを有効にするには、`service` の `replicas` を次の設定に置き換えます。

```yaml
service:
  replica_policy:
    min_replicas: 2
    max_replicas: 4
    target_qps_per_replica: 2
```

これにより、各レプリカの QPS が 2 を超えたときにサービスがスケールアップします。

??? code "YAML"

    ```yaml
    service:
      replica_policy:
        min_replicas: 2
        max_replicas: 4
        target_qps_per_replica: 2
      # An actual request for readiness probe.
      readiness_probe:
        path: /v1/chat/completions
        post_data:
          model: $MODEL_NAME
          messages:
            - role: user
              content: Hello! What is your name?
          max_completion_tokens: 1

    resources:
      accelerators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # We can use cheaper accelerators for 8B model.
      use_spot: True
      disk_size: 512  # Ensure model checkpoints can fit.
      disk_tier: best
      ports: 8081  # Expose to internet traffic.

    envs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      HF_TOKEN: <your-huggingface-token>  # Change to your own huggingface token, or use --env to pass.

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      pip install vllm==0.4.0.post1
      # Install Gradio for web UI.
      pip install gradio openai
      pip install flash-attn==2.5.7

    run: |
      conda activate vllm
      echo 'Starting vllm api server...'
      vllm serve $MODEL_NAME \
        --port 8081 \
        --trust-remote-code \
        --tensor-parallel-size $SKYPILOT_NUM_GPUS_PER_NODE \
        2>&1 | tee api_server.log
    ```

新しい設定でサービスを更新するには次のようにします。

```bash
HF_TOKEN="your-huggingface-token" sky serve update vllm serving.yaml --env HF_TOKEN
```

サービスを停止するには次のようにします。

```bash
sky serve down vllm
```

### **任意**: エンドポイントに GUI を接続する { #optional-connect-a-gui-to-the-endpoint }

別途 GUI のフロントエンドから Llama-3 のサービスにアクセスすることもできます。GUI に送られたユーザーのリクエストは、レプリカ間で負荷分散されます。

??? code "YAML"

    ```yaml
    envs:
      MODEL_NAME: meta-llama/Meta-Llama-3-8B-Instruct
      ENDPOINT: x.x.x.x:3031 # Address of the API server running vllm.

    resources:
      cpus: 2

    setup: |
      conda create -n vllm python=3.10 -y
      conda activate vllm

      # Install Gradio for web UI.
      pip install gradio openai

    run: |
      conda activate vllm
      export PATH=$PATH:/sbin

      echo 'Starting gradio server...'
      git clone https://github.com/vllm-project/vllm.git || true
      python vllm/examples/applications/api_client/gradio_openai_chatbot_webserver.py \
        -m $MODEL_NAME \
        --port 8811 \
        --model-url http://$ENDPOINT/v1 \
        --stop-token-ids 128009,128001 | tee ~/gradio.log
    ```

1. チャットの Web UI を起動します。

    ```bash
    sky launch \
      -c gui ./gui.yaml \
      --env ENDPOINT=$(sky serve status --endpoint vllm)
    ```

2. 返された gradio のリンクから GUI にアクセスできます。

    ```console
    | INFO | stdout | Running on public URL: https://6141e84201ce0bb4ed.gradio.live
    ```
