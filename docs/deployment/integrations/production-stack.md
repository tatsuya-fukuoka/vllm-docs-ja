# Production stack { #production-stack }

Kubernetes 上への vLLM のデプロイは、機械学習モデルをスケーラブルかつ効率的にサービングする方法です。このガイドでは、[vLLM production stack](https://github.com/vllm-project/production-stack) を使ったデプロイ手順を説明します。バークレー校とシカゴ大学の共同研究から生まれた vLLM production stack は、[vLLM プロジェクト](https://github.com/vllm-project)配下で公式に公開されている本番向けに最適化されたコードベースで、次の特徴を持つ LLM のデプロイを想定しています。

* **上流 vLLM との互換性** – 上流の vLLM のコードを変更せずにラップします。
* **使いやすさ** – Helm チャートによる簡単なデプロイと、Grafana ダッシュボードによる可観測性。
* **高い性能** – 複数モデルのサポート、モデル / プレフィックスを考慮したルーティング、vLLM の高速な起動、[LMCache](https://github.com/LMCache/LMCache) による KV キャッシュのオフロード（vLLM では `--kv-offloading-backend lmcache` で接続します。[LMCache の例](https://github.com/vllm-project/vllm/tree/main/examples/disaggregated/lmcache)と [docs.lmcache.ai](https://docs.lmcache.ai) を参照）など、LLM のワークロード向けに最適化されています。

Kubernetes が初めてでも心配ありません。vLLM production stack の[リポジトリ](https://github.com/vllm-project/production-stack)には、**4 分**で環境を用意して始められる[手順ガイド](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)と[短い動画](https://www.youtube.com/watch?v=EsTJbQtzj0g)があります。

## 前提条件 { #pre-requisite }

GPU を備えた Kubernetes 環境が稼働していることを確認してください（ベアメタルの GPU マシンに Kubernetes 環境を構築する手順は[このチュートリアル](https://github.com/vllm-project/production-stack/blob/main/tutorials/00-install-kubernetes-env.md)を参照）。

## vLLM production stack を使ったデプロイ { #deployment-using-vllm-production-stack }

標準的な vLLM production stack は Helm チャートでインストールします。GPU サーバーへの Helm のインストールには、この [bash スクリプト](https://github.com/vllm-project/production-stack/blob/main/utils/install-helm.sh)を利用できます。

vLLM production stack をインストールするには、手元の環境で次のコマンドを実行します。

```bash
sudo helm repo add vllm https://vllm-project.github.io/production-stack
sudo helm install vllm vllm/vllm-stack -f tutorials/assets/values-01-minimal-example.yaml
```

これにより、小さな LLM（Facebook の opt-125M モデル）を動かす `vllm` という名前の vLLM production stack ベースのデプロイが作成されます。

### インストールの確認 { #validate-installation }

次のコマンドでデプロイの状態を確認します。

```bash
sudo kubectl get pods
```

`vllm` デプロイの Pod が `Running` 状態に遷移するのが確認できます。

```text
NAME                                           READY   STATUS    RESTARTS   AGE
vllm-deployment-router-859d8fb668-2x2b7        1/1     Running   0          2m38s
vllm-opt125m-deployment-vllm-84dfc9bd7-vb9bs   1/1     Running   0          2m38s
```

!!! note
    コンテナが Docker イメージと LLM の重みをダウンロードするまで、しばらく時間がかかることがあります。

### スタックにクエリを送る { #send-a-query-to-the-stack }

`vllm-router-service` のポートをホストマシンに転送します。

```bash
sudo kubectl port-forward svc/vllm-router-service 30080:80
```

そのうえで、OpenAI 互換 API にクエリを送って利用可能なモデルを確認できます。

```bash
curl -o- http://localhost:30080/v1/models
```

??? console "出力"

    ```json
    {
      "object": "list",
      "data": [
        {
          "id": "facebook/opt-125m",
          "object": "model",
          "created": 1737428424,
          "owned_by": "vllm",
          "root": null
        }
      ]
    }
    ```

実際にチャットのリクエストを送るには、OpenAI の `/completion` エンドポイントに curl でリクエストします。

```bash
curl -X POST http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time,",
    "max_tokens": 10
  }'
```

??? console "出力"

    ```json
    {
      "id": "completion-id",
      "object": "text_completion",
      "created": 1737428424,
      "model": "facebook/opt-125m",
      "choices": [
        {
          "text": " there was a brave knight who...",
          "index": 0,
          "finish_reason": "length"
        }
      ]
    }
    ```

### アンインストール { #uninstall }

デプロイを削除するには次を実行します。

```bash
sudo helm uninstall vllm
```

---

### （応用）vLLM production stack の設定 { #advanced-configuring-vllm-production-stack }

vLLM production stack の中心的な設定は YAML で管理します。上記のインストールで使った設定例は次のとおりです。

??? code "YAML"

    ```yaml
    servingEngineSpec:
      runtimeClassName: ""
      modelSpec:
      - name: "opt125m"
        repository: "vllm/vllm-openai"
        tag: "latest"
        modelURL: "facebook/opt-125m"

        replicaCount: 1

        requestCPU: 6
        requestMemory: "16Gi"
        requestGPU: 1

        pvcStorage: "10Gi"
    ```

この YAML 設定の各項目:

* **`modelSpec`** に含まれる項目:
    * `name`: モデルに付ける任意の呼び名。
    * `repository`: vLLM の Docker リポジトリ。
    * `tag`: Docker イメージのタグ。
    * `modelURL`: 使用したい LLM のモデル。
* **`replicaCount`**: レプリカ数。
* **`requestCPU` と `requestMemory`**: Pod に要求する CPU とメモリのリソース量。
* **`requestGPU`**: 必要な GPU の数。
* **`pvcStorage`**: モデル用に確保する永続ストレージ。

!!! note
    Pod を 2 つ構成したい場合は、この [YAML ファイル](https://github.com/vllm-project/production-stack/blob/main/tutorials/assets/values-01-2pods-minimal-example.yaml)を参照してください。

!!! tip
    vLLM production stack には、CPU へのオフロードや多様なルーティングアルゴリズムなど、さらに多くの機能があります。詳細は[例とチュートリアル](https://github.com/vllm-project/production-stack/tree/main/tutorials)や[リポジトリ](https://github.com/vllm-project/production-stack)を参照してください。
