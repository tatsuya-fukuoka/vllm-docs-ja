# Kthena { #kthena }

[**Kthena**](https://github.com/volcano-sh/kthena) は Kubernetes ネイティブの LLM 推論プラットフォームで、大規模言語モデルを本番環境でデプロイ・運用する方法を刷新します。宣言的なモデルライフサイクル管理を軸に設計されています。

このガイドでは、本番品質の**複数ノード構成の vLLM** サービスを Kubernetes 上にデプロイする方法を説明します。

手順は次のとおりです。

- 必要なコンポーネント（Kthena と Volcano）をインストールする。
- Kthena の `ModelServing` CR で複数ノードの vLLM モデルをデプロイする。
- デプロイを検証する。

---

## 1. 前提条件 { #1-prerequisites }

必要なもの:

- **GPU ノード**を備えた Kubernetes クラスタ。
- cluster-admin 相当の権限を持つ `kubectl` のアクセス。
- ギャングスケジューリングのための **Volcano** のインストール。
- `ModelServing` CRD が利用できる状態での **Kthena** のインストール。
- Hugging Face Hub からモデルを読み込む場合は、有効な **Hugging Face のトークン**。

### 1.1 Volcano のインストール { #11-install-volcano }

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

これにより、Kthena が使うギャングスケジューリングとネットワークトポロジの機能が利用できるようになります。

### 1.2 Kthena のインストール { #12-install-kthena }

```bash
helm install kthena oci://ghcr.io/volcano-sh/charts/kthena --version v0.1.0 --namespace kthena-system --create-namespace
```

- `kthena-system` 名前空間が作成されます。
- `ModelServing` を含む Kthena のコントローラーと CRD がインストールされ、正常に動作します。

確認します。

```bash
kubectl get crd | grep modelserving
```

次のような出力が得られるはずです。

```text
modelservings.workload.serving.volcano.sh   ...
```

---

## 2. 複数ノード vLLM の `ModelServing` の例 { #2-the-multi-node-vllm-modelserving-example }

Kthena は、**Llama を動かす複数ノードの vLLM クラスタ**をデプロイするマニフェストの例を提供しています。概念的には vLLM production stack の Helm デプロイと同等ですが、`ModelServing` で表現されています。

例（`llama-multinode`）を簡略化すると次のようになります。

- `spec.replicas: 1` – `ServingGroup` が 1 つ（論理的なモデルデプロイが 1 つ）。
- `roles`:
    - `entryTemplate` – 次を実行する**リーダー** Pod を定義します。
        - vLLM の**複数ノードのクラスタ初期化スクリプト**。
        - vLLM の **OpenAI 互換 API サーバー**。
    - `workerTemplate` – リーダーの Ray クラスタ（Ray バックエンド）または同じ分散プロセスグループ（multiprocessing バックエンド）に参加する**ワーカー** Pod を定義します。

例の YAML の要点:

イメージ: `vllm/vllm-openai:latest`（上流の vLLM イメージと同じもの）。
コマンド:

??? code "Yaml"
    === "Multiprocessing（既定）"
        リーダー:

        ```yaml
        command:
          - sh
          - -c
          - >
            vllm serve meta-llama/Llama-3.1-405B-Instruct
              --tensor-parallel-size 8
              --pipeline-parallel-size 2
              --nnodes=2
              --node-rank=0
              --master-addr=$(ENTRY_ADDRESS)
              --port 8080
        ```

        ワーカー:

        ```yaml
        command:
          - sh
          - -c
          - >
            vllm serve meta-llama/Llama-3.1-405B-Instruct
              --tensor-parallel-size 8
              --pipeline-parallel-size 2
              --nnodes=2
              --node-rank=1
              --master-addr=$(ENTRY_ADDRESS)
              --headless
        ```

    === "Ray"
        リーダー:

        ```yaml
        command:
          - sh
          - -c
          - >
            bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh
            leader --ray_cluster_size=2; python3 -m
            vllm.entrypoints.openai.api_server --port 8080 --model
            meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8
            --pipeline-parallel-size 2
        ```

        ワーカー:

        ```yaml
        command:
          - sh
          - -c
          - >
            bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh
            worker --ray_address=$(ENTRY_ADDRESS)
        ```

---

## 3. Kthena で複数ノードの llama vLLM をデプロイする { #3-deploying-multi-node-llama-vllm-via-kthena }

### 3.1 マニフェストの準備 { #31-prepare-the-manifest }

**推奨**: 環境変数に直接書くのではなく Secret を使ってください。

```bash
kubectl create secret generic hf-token \
  -n default \
  --from-literal=HUGGING_FACE_HUB_TOKEN='<your-token>'
```

### 3.2 `ModelServing` の適用 { #32-apply-the-modelserving }

次のいずれかのマニフェストを `modelserving.yaml` として保存します。

??? code "modelserving.yaml"
    === "Multiprocessing（既定）"
        ```yaml
        apiVersion: workload.serving.volcano.sh/v1alpha1
        kind: ModelServing
        metadata:
          name: llama-multinode
          namespace: default
        spec:
          schedulerName: volcano
          replicas: 1  # group replicas
          template:
            restartGracePeriodSeconds: 60
            gangPolicy:
              minRoleReplicas:
                405b: 1
            roles:
              - name: 405b
                replicas: 2
                entryTemplate:
                  spec:
                    containers:
                      - name: leader
                        image: vllm/vllm-openai:latest
                        env:
                          - name: HUGGING_FACE_HUB_TOKEN
                            valueFrom:
                              secretKeyRef:
                                name: hf-token
                                key: HUGGING_FACE_HUB_TOKEN
                        command:
                          - sh
                          - -c
                          - "vllm serve meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 0 --master-addr $(ENTRY_ADDRESS) --distributed-executor-backend mp --port 8080"
                        resources:
                          limits:
                            nvidia.com/gpu: "8"
                            memory: 1124Gi
                            ephemeral-storage: 800Gi
                          requests:
                            ephemeral-storage: 800Gi
                            cpu: 125
                        ports:
                          - containerPort: 8080
                        readinessProbe:
                          tcpSocket:
                            port: 8080
                          initialDelaySeconds: 15
                          periodSeconds: 10
                        volumeMounts:
                          - mountPath: /dev/shm
                            name: dshm
                    volumes:
                    - name: dshm
                      emptyDir:
                        medium: Memory
                        sizeLimit: 15Gi
                workerReplicas: 1
                workerTemplate:
                  spec:
                    containers:
                      - name: worker
                        image: vllm/vllm-openai:latest
                        command:
                          - sh
                          - -c
                          - "vllm serve meta-llama/Llama-3.1-405B-Instruct --tensor-parallel-size 8 --pipeline-parallel-size 2 --nnodes 2 --node-rank 1 --master-addr $(ENTRY_ADDRESS) --distributed-executor-backend mp --headless"
                        resources:
                          limits:
                            nvidia.com/gpu: "8"
                            memory: 1124Gi
                            ephemeral-storage: 800Gi
                          requests:
                            ephemeral-storage: 800Gi
                            cpu: 125
                        env:
                          - name: HUGGING_FACE_HUB_TOKEN
                            valueFrom:
                              secretKeyRef:
                                name: hf-token
                                key: HUGGING_FACE_HUB_TOKEN
                        volumeMounts:
                          - mountPath: /dev/shm
                            name: dshm
                    volumes:
                    - name: dshm
                      emptyDir:
                        medium: Memory
                        sizeLimit: 15Gi
        ```

    === "Ray"
        ```yaml
        apiVersion: workload.serving.volcano.sh/v1alpha1
        kind: ModelServing
        metadata:
          name: llama-multinode
          namespace: default
        spec:
          schedulerName: volcano
          replicas: 1  # group replicas
          template:
            restartGracePeriodSeconds: 60
            gangPolicy:
              minRoleReplicas:
                405b: 1
            roles:
              - name: 405b
                replicas: 2
                entryTemplate:
                  spec:
                    containers:
                      - name: leader
                        image: vllm/vllm-openai:latest
                        env:
                          - name: HUGGING_FACE_HUB_TOKEN
                            valueFrom:
                              secretKeyRef:
                                name: hf-token
                                key: HUGGING_FACE_HUB_TOKEN
                        command:
                          - sh
                          - -c
                          - "bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh leader --ray_cluster_size=2;
                            vllm serve meta-llama/Llama-3.1-405B-Instruct --port 8080 --tensor-parallel-size 8 --pipeline-parallel-size 2"
                        resources:
                          limits:
                            nvidia.com/gpu: "8"
                            memory: 1124Gi
                            ephemeral-storage: 800Gi
                          requests:
                            ephemeral-storage: 800Gi
                            cpu: 125
                        ports:
                          - containerPort: 8080
                        readinessProbe:
                          tcpSocket:
                            port: 8080
                          initialDelaySeconds: 15
                          periodSeconds: 10
                        volumeMounts:
                          - mountPath: /dev/shm
                            name: dshm
                    volumes:
                    - name: dshm
                      emptyDir:
                        medium: Memory
                        sizeLimit: 15Gi
                workerReplicas: 1
                workerTemplate:
                  spec:
                    containers:
                      - name: worker
                        image: vllm/vllm-openai:latest
                        command:
                          - sh
                          - -c
                          - "bash /vllm-workspace/examples/ray_serving/multi-node-serving.sh worker --ray_address=$(ENTRY_ADDRESS)"
                        resources:
                          limits:
                            nvidia.com/gpu: "8"
                            memory: 1124Gi
                            ephemeral-storage: 800Gi
                          requests:
                            ephemeral-storage: 800Gi
                            cpu: 125
                        env:
                          - name: HUGGING_FACE_HUB_TOKEN
                            valueFrom:
                              secretKeyRef:
                                name: hf-token
                                key: HUGGING_FACE_HUB_TOKEN
                        volumeMounts:
                          - mountPath: /dev/shm
                            name: dshm
                    volumes:
                    - name: dshm
                      emptyDir:
                        medium: Memory
                        sizeLimit: 15Gi
        ```

```bash
kubectl apply -f modelserving.yaml
```

Kthena は次を行います。

- `ModelServing` オブジェクトを作成します。
- Volcano のギャングスケジューリング用に `PodGroup` を導出します。
- `ServingGroup` と `Role` ごとにリーダーとワーカーの Pod を作成します。

---

## 4. デプロイの確認 { #4-verifying-the-deployment }

### 4.1 ModelServing の状態を確認する { #41-check-modelserving-status }

Kthena のドキュメントにあるスニペットを使います。

```bash
kubectl get modelserving -oyaml | grep status -A 10
```

次のような出力が得られるはずです。

```yaml
status:
  availableReplicas: 1
  conditions:
    - type: Available
      status: "True"
      reason: AllGroupsReady
      message: All Serving groups are ready
    - type: Progressing
      status: "False"
      ...
  replicas: 1
  updatedReplicas: 1
```

### 4.2 Pod を確認する { #42-check-pods }

デプロイの Pod を一覧表示します。

```bash
kubectl get pod -owide -l modelserving.volcano.sh/name=llama-multinode
```

出力例（ドキュメントより）:

```text
NAMESPACE   NAME                          READY   STATUS    RESTARTS   AGE   IP            NODE           ...
default     llama-multinode-0-405b-0-0    1/1     Running   0          15m   10.244.0.56   192.168.5.12   ...
default     llama-multinode-0-405b-0-1    1/1     Running   0          15m   10.244.0.58   192.168.5.43   ...
default     llama-multinode-0-405b-1-0    1/1     Running   0          15m   10.244.0.57   192.168.5.58   ...
default     llama-multinode-0-405b-1-1    1/1     Running   0          15m   10.244.0.53   192.168.5.36   ...
```

Pod 名のパターン:

- `llama-multinode-<group-idx>-<role-name>-<replica-idx>-<ordinal>`.

最初の数字が `ServingGroup`、2 つ目（`405b`）が `Role` を表します。残りの添字は、その Role 内での Pod を識別します。

---

## 6. vLLM の OpenAI 互換 API にアクセスする { #6-accessing-the-vllm-openai-compatible-api }

Service でエントリを公開します。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-multinode-openai
  namespace: default
spec:
  selector:
    modelserving.volcano.sh/name: llama-multinode
    modelserving.volcano.sh/entry: "true"
    # optionally further narrow to leader role if you label it
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

手元のマシンからポートフォワードします。

```bash
kubectl port-forward svc/llama-multinode-openai 30080:80 -n default
```

そのうえで:

- モデルの一覧を取得します。

  ```bash
  curl -s http://localhost:30080/v1/models
  ```

- completion のリクエストを送ります（vLLM production stack のドキュメントと同様）。

  ```bash
  curl -X POST http://localhost:30080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "meta-llama/Llama-3.1-405B-Instruct",
      "prompt": "Once upon a time,",
      "max_tokens": 10
    }'
  ```

vLLM から OpenAI 形式のレスポンスが返るはずです。

---

## 7. 後片付け { #7-clean-up }

デプロイとその関連リソースを削除するには次のようにします。

```bash
kubectl delete modelserving llama-multinode -n default
```

スタック全体を使い終えた場合は次のようにします。

```bash
helm uninstall kthena -n kthena-system   # or your Kthena release name
helm uninstall volcano -n volcano-system
```
