# Kubernetes を使う { #using-kubernetes }

Kubernetes 上への vLLM のデプロイは、機械学習モデルをスケーラブルかつ効率的にサービングする方法です。このガイドでは、素の Kubernetes を使って vLLM をデプロイする手順を説明します。

- [CPU でのデプロイ](#deployment-with-cpus)
- [GPU でのデプロイ](#deployment-with-gpus)
- [gRPC でのサービング](#serving-with-grpc)
- [トラブルシューティング](#troubleshooting)
    - [Startup Probe / Readiness Probe の失敗（コンテナログに "KeyboardInterrupt: terminated" が出る）](#startup-probe-or-readiness-probe-failure-container-log-contains-keyboardinterrupt-terminated)
- [まとめ](#conclusion)

次のいずれかを使って vLLM を Kubernetes にデプロイすることもできます。

- [Helm](frameworks/helm.md)
- [NVIDIA Dynamo](integrations/dynamo.md)
- [InftyAI/llmaz](integrations/llmaz.md)
- [llm-d](integrations/llm-d.md)
- [KAITO](integrations/kaito.md)
- [KServe](integrations/kserve.md)
- [Kthena](integrations/kthena.md)
- [KubeRay](integrations/kuberay.md)
- [kubernetes-sigs/lws](frameworks/lws.md)
- [meta-llama/llama-stack](integrations/llamastack.md)
- [substratusai/kubeai](integrations/kubeai.md)
- [vllm-project/AIBrix](integrations/aibrix.md)
- [vllm-project/production-stack](integrations/production-stack.md)

## CPU でのデプロイ { #deployment-with-cpus }

!!! note
    ここでの CPU の使用はデモとテストが目的であり、性能は GPU と同等にはなりません。

まず、Hugging Face のモデルをダウンロード・保存するための Kubernetes PVC と Secret を作成します。

??? console "Config"

    ```bash
    cat <<EOF |kubectl apply -f -
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
      name: vllm-models
    spec:
      accessModes:
        - ReadWriteOnce
      volumeMode: Filesystem
      resources:
        requests:
          storage: 50Gi
    ---
    apiVersion: v1
    kind: Secret
    metadata:
      name: hf-token-secret
    type: Opaque
    stringData:
      token: "REPLACE_WITH_TOKEN"
    EOF
    ```

ここで `token` フィールドには **Hugging Face のアクセストークン**を格納します。トークンの発行方法は
[Hugging Face のドキュメント](https://huggingface.co/docs/hub/en/security-tokens)を参照してください。

次に、Kubernetes の Deployment と Service として vLLM サーバーを起動します。

vLLM のイメージは、プロセッサのアーキテクチャに合わせて指定してください。

??? console "Config"

    ```bash
    VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo:latest       # use this for x86_64
    VLLM_IMAGE=public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:latest # use this for arm64
    cat <<EOF |kubectl apply -f -
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: vllm-server
    spec:
      replicas: 1
      selector:
        matchLabels:
          app.kubernetes.io/name: vllm
      template:
        metadata:
          labels:
            app.kubernetes.io/name: vllm
        spec:
          containers:
          - name: vllm
            image: $VLLM_IMAGE
            command: ["/bin/sh", "-c"]
            args: [
              "vllm serve meta-llama/Llama-3.2-1B-Instruct"
            ]
            env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            ports:
              - containerPort: 8000
            volumeMounts:
              - name: llama-storage
                mountPath: /root/.cache/huggingface
          volumes:
          - name: llama-storage
            persistentVolumeClaim:
              claimName: vllm-models
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: vllm-server
    spec:
      selector:
        app.kubernetes.io/name: vllm
      ports:
      - protocol: TCP
        port: 8000
        targetPort: 8000
      type: ClusterIP
    EOF
    ```

ログから vLLM サーバーが正常に起動したことを確認できます（モデルのダウンロードに数分かかることがあります）。

```bash
kubectl logs -l app.kubernetes.io/name=vllm
...
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## GPU でのデプロイ { #deployment-with-gpus }

**前提条件**: [GPU を備えた Kubernetes クラスタ](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)が稼働していること。

1. vLLM 用の PVC・Secret・Deployment を作成する

      PVC はモデルのキャッシュを保存するために使います。任意であり、hostPath や他のストレージオプションでも構いません。

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: v1
      kind: PersistentVolumeClaim
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        accessModes:
        - ReadWriteOnce
        resources:
          requests:
            storage: 50Gi
        storageClassName: default
        volumeMode: Filesystem
      ```

      </details>

      Secret も任意で、アクセス制限のある gated モデルを使う場合にのみ必要です。gated モデルを使わない場合はこの手順を省略できます。

      ```yaml
      apiVersion: v1
      kind: Secret
      metadata:
        name: hf-token-secret
        namespace: default
      type: Opaque
      stringData:
        token: "REPLACE_WITH_TOKEN"
      ```
  
      続いて、モデルサーバーを動かすための vLLM の Deployment ファイルを作成します。次の例では `Mistral-7B-Instruct-v0.3` モデルをデプロイします。

      NVIDIA GPU と AMD GPU の 2 つの例を示します。

      NVIDIA GPU の場合:

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM needs to access the host's shared memory for tensor parallel inference.
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "2Gi"
            containers:
            - name: mistral-7b
              image: vllm/vllm-openai:latest
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --trust-remote-code --enable-chunked-prefill --max-num-batched-tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  nvidia.com/gpu: "1"
                requests:
                  cpu: "2"
                  memory: 6G
                  nvidia.com/gpu: "1"
              volumeMounts:
              - mountPath: /root/.cache/huggingface
                name: cache-volume
              - name: shm
                mountPath: /dev/shm
              livenessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 10
              readinessProbe:
                httpGet:
                  path: /health
                  port: 8000
                initialDelaySeconds: 60
                periodSeconds: 5
      ```

      </details>

      AMD GPU:

      MI300X などの AMD ROCm GPU を使う場合は、以下の `deployment.yaml` を参考にしてください。

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: mistral-7b
        namespace: default
        labels:
          app: mistral-7b
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: mistral-7b
        template:
          metadata:
            labels:
              app: mistral-7b
          spec:
            volumes:
            # PVC
            - name: cache-volume
              persistentVolumeClaim:
                claimName: mistral-7b
            # vLLM needs to access the host's shared memory for tensor parallel inference.
            - name: shm
              emptyDir:
                medium: Memory
                sizeLimit: "8Gi"
            hostNetwork: true
            hostIPC: true
            containers:
            - name: mistral-7b
              image: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4
              securityContext:
                seccompProfile:
                  type: Unconfined
                runAsGroup: 44
                capabilities:
                  add:
                  - SYS_PTRACE
              command: ["/bin/sh", "-c"]
              args: [
                "vllm serve mistralai/Mistral-7B-v0.3 --port 8000 --trust-remote-code --enable-chunked-prefill --max-num-batched-tokens 1024"
              ]
              env:
              - name: HF_TOKEN
                valueFrom:
                  secretKeyRef:
                    name: hf-token-secret
                    key: token
              ports:
              - containerPort: 8000
              resources:
                limits:
                  cpu: "10"
                  memory: 20G
                  amd.com/gpu: "1"
                requests:
                  cpu: "6"
                  memory: 6G
                  amd.com/gpu: "1"
              volumeMounts:
              - name: cache-volume
                mountPath: /root/.cache/huggingface
              - name: shm
                mountPath: /dev/shm
      ```

      </details>

      手順とサンプルの yaml を含む完全な例は <https://github.com/ROCm/k8s-device-plugin/tree/master/example/vllm-serve> から入手できます。

2. vLLM 用の Kubernetes Service を作成する

      次に、`mistral-7b` の Deployment を公開するための Kubernetes Service ファイルを作成します。

      <details>
      <summary>Yaml</summary>

      ```yaml
      apiVersion: v1
      kind: Service
      metadata:
        name: mistral-7b
        namespace: default
      spec:
        ports:
        - name: http-mistral-7b
          port: 80
          protocol: TCP
          targetPort: 8000
        # The label selector should match the deployment labels & it is useful for prefix caching feature
        selector:
          app: mistral-7b
        sessionAffinity: None
        type: ClusterIP
      ```

      </details>

3. デプロイして動作確認する

      `kubectl apply -f <filename>` で Deployment と Service の設定を適用します。

      ```bash
      kubectl apply -f deployment.yaml
      kubectl apply -f service.yaml
      ```

      デプロイを確認するには、次の `curl` コマンドを実行します。

      ```bash
      curl http://mistral-7b.default.svc.cluster.local/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
              "model": "mistralai/Mistral-7B-Instruct-v0.3",
              "prompt": "San Francisco is a",
              "max_tokens": 7,
              "temperature": 0
            }'
      ```

      Service が正しくデプロイされていれば、vLLM のモデルからレスポンスが返ります。

## gRPC でのサービング { #serving-with-grpc }

`--grpc` フラグを指定すると、vLLM は HTTP ではなく gRPC でモデルをサービングできます。これにはオプションの gRPC 依存パッケージが必要です。

```bash
pip install vllm[grpc]
```

`--grpc` を使うと、サーバーは標準の [gRPC Health Checking Protocol](https://github.com/grpc/grpc/blob/master/doc/health-checking.md)（`grpc.health.v1.Health`）を公開します。これは Kubernetes の [ネイティブ gRPC プローブ](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-grpc-liveness-probe)（Kubernetes 1.24 以降）と連携します。

gRPC でデプロイするには、`vllm serve` コマンドに `--grpc` を追加し、`httpGet` のプローブを `grpc` のプローブに置き換えます。

```yaml
containers:
- name: mistral-7b
  image: vllm/vllm-openai:latest
  command: ["/bin/sh", "-c"]
  args: [
    "pip install vllm[grpc] && vllm serve mistralai/Mistral-7B-Instruct-v0.3 --grpc --port 50051 --trust-remote-code"
  ]
  ports:
  - containerPort: 50051
  livenessProbe:
    grpc:
      port: 50051
    initialDelaySeconds: 120
    periodSeconds: 10
  readinessProbe:
    grpc:
      port: 50051
    initialDelaySeconds: 120
    periodSeconds: 5
```

!!! note
    gRPC のヘルスサービスは、プローブのたびにエンジンの状態を確認します。エンジンが異常な場合やサーバーが停止中の場合、プローブは `NOT_SERVING` を返します。

`grpcurl` を使って手動でヘルスサービスを確認することもできます。

```bash
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

## トラブルシューティング { #troubleshooting }

### Startup Probe / Readiness Probe の失敗（コンテナログに "KeyboardInterrupt: terminated" が出る） { #startup-probe-or-readiness-probe-failure-container-log-contains-keyboardinterrupt-terminated }

startup / readiness プローブの failureThreshold がサーバーの起動時間に対して小さすぎると、Kubernetes のスケジューラがコンテナを停止します。これが起きたときの主な兆候は次のとおりです。

1. コンテナのログに "KeyboardInterrupt: terminated" が出力される
2. `kubectl get events` に `Container $NAME failed startup probe, will be restarted` というメッセージが出る

対処としては、failureThreshold を大きくしてモデルサーバーの起動に時間を与えてください。適切な値は、マニフェストからプローブを外したうえで、モデルサーバーがサービス可能になるまでの時間を計測すると判断できます。

## まとめ { #conclusion }

Kubernetes 上に vLLM をデプロイすると、GPU リソースを活かして ML モデルを効率的にスケール・管理できます。上記の手順に従えば、Kubernetes クラスタ内で vLLM のデプロイを構築してテストできるはずです。問題や改善案があれば、ぜひドキュメントへのコントリビューションをご検討ください。
