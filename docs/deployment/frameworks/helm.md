# Helm { #helm }

Kubernetes に vLLM をデプロイするための Helm チャートです。

Helm は Kubernetes のパッケージマネージャーで、Kubernetes 上への vLLM アプリケーションのデプロイを自動化できます。変数の値を上書きすることで、同じ構成を異なる設定で複数の名前空間にデプロイできます。

このガイドでは、前提条件、Helm でのインストール手順、アーキテクチャと values ファイルの説明を含め、Helm を使った vLLM のデプロイ手順を説明します。

## 前提条件 { #prerequisites }

始める前に、次が用意されていることを確認してください。

- 稼働中の Kubernetes クラスタ
- NVIDIA Kubernetes Device Plugin (`k8s-device-plugin`): [https://github.com/NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) から入手できます
- クラスタ内で利用可能な GPU リソース
- （任意）モデルの自動ダウンロードを使う場合は、モデルの重みを置いた S3 バケットなどのストレージ

## チャートのインストール { #installing-the-chart }

このガイドでは [examples/deployment/chart-helm](../../../examples/deployment/chart-helm) の Helm チャートを使います。

リリース名 `test-vllm` でチャートをインストールするには次のようにします。

```bash
helm upgrade --install --create-namespace \
  --namespace=ns-vllm test-vllm . \
  -f values.yaml \
  --set secrets.s3endpoint=$ACCESS_POINT \
  --set secrets.s3bucketname=$BUCKET \
  --set secrets.s3accesskeyid=$ACCESS_KEY \
  --set secrets.s3accesskey=$SECRET_KEY
```

## チャートのアンインストール { #uninstalling-the-chart }

`test-vllm` のデプロイをアンインストールするには次のようにします。

```bash
helm uninstall test-vllm --namespace=ns-vllm
```

このコマンドは、チャートに関連するすべての Kubernetes コンポーネント（**永続ボリュームを含む**）を削除し、
リリースを削除します。

## アーキテクチャ { #architecture }

![helm deployment architecture](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/deployment/architecture_helm_deployment.png)

## 設定値 { #values }

次の表は、`values.yaml` で設定できるチャートのパラメータです。

| キー | 型 | 既定値 | 説明 |
| --- | ---- | ------- | ----------- |
| autoscaling | object | {"enabled":false,"maxReplicas":100,"minReplicas":1,"targetCPUUtilizationPercentage":80} | Autoscaling configuration |
| autoscaling.enabled | bool | false | Enable autoscaling |
| autoscaling.maxReplicas | int | 100 | Maximum replicas |
| autoscaling.minReplicas | int | 1 | Minimum replicas |
| autoscaling.targetCPUUtilizationPercentage | int | 80 | Target CPU utilization for autoscaling |
| configs | object | {} | Configmap |
| containerPort | int | 8000 | Container port |
| customObjects | list | [] | Custom Objects configuration |
| deploymentStrategy | object | {} | Deployment strategy configuration |
| externalConfigs | list | [] | External configuration |
| extraContainers | list | [] | Additional containers configuration |
| extraInit | object | {"modelDownload":{"enabled":true},"initContainers":[],"pvcStorage":"1Gi"} | Additional configuration for init containers |
| extraInit.modelDownload | object | {"enabled":true} | Model download functionality configuration |
| extraInit.modelDownload.enabled | bool | true | Enable automatic model download job and wait container |
| extraInit.modelDownload.image | object | {"repository":"amazon/aws-cli","tag":"2.6.4","pullPolicy":"IfNotPresent"} | Image for model download operations |
| extraInit.modelDownload.waitContainer | object | {} | Wait container configuration (command, args, env) |
| extraInit.modelDownload.downloadJob | object | {} | Download job configuration (command, args, env) |
| extraInit.initContainers | list | [] | Custom init containers (appended after model download if enabled) |
| extraInit.pvcStorage | string | "1Gi" | Storage size for the PVC |
| extraInit.s3modelpath | string | "relative_s3_model_path/opt-125m" | (Optional) Path of the model on S3 |
| extraInit.awsEc2MetadataDisabled | bool | true | (Optional) Disable AWS EC2 metadata service |
| extraPorts | list | [] | Additional ports configuration |
| gpuModels | list | ["TYPE_GPU_USED"] | Type of gpu used |
| image | object | {"command":["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"],"repository":"vllm/vllm-openai","tag":"latest"} | Image configuration |
| image.command | list | ["vllm","serve","/data/","--served-model-name","opt-125m","--host","0.0.0.0","--port","8000"] | Container launch command |
| image.repository | string | "vllm/vllm-openai" | Image repository |
| image.tag | string | "latest" | Image tag |
| livenessProbe | object | {"failureThreshold":3,"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":15,"periodSeconds":10} | Liveness probe configuration |
| livenessProbe.failureThreshold | int | 3 | Number of times after which if a probe fails in a row, Kubernetes considers that the overall check has failed: the container is not alive |
| livenessProbe.httpGet | object | {"path":"/health","port":8000} | Configuration of the kubelet http request on the server |
| livenessProbe.httpGet.path | string | "/health" | Path to access on the HTTP server |
| livenessProbe.httpGet.port | int | 8000 | Name or number of the port to access on the container, on which the server is listening |
| livenessProbe.initialDelaySeconds | int | 15 | Number of seconds after the container has started before liveness probe is initiated |
| livenessProbe.periodSeconds | int | 10 | How often (in seconds) to perform the liveness probe |
| maxUnavailablePodDisruptionBudget | string | "" | Disruption Budget Configuration |
| readinessProbe | object | {"failureThreshold":3,"httpGet":{"path":"/health","port":8000},"initialDelaySeconds":5,"periodSeconds":5} | Readiness probe configuration |
| readinessProbe.failureThreshold | int | 3 | Number of times after which if a probe fails in a row, Kubernetes considers that the overall check has failed: the container is not ready |
| readinessProbe.httpGet | object | {"path":"/health","port":8000} | Configuration of the kubelet http request on the server |
| readinessProbe.httpGet.path | string | "/health" | Path to access on the HTTP server |
| readinessProbe.httpGet.port | int | 8000 | Name or number of the port to access on the container, on which the server is listening |
| readinessProbe.initialDelaySeconds | int | 5 | Number of seconds after the container has started before readiness probe is initiated |
| readinessProbe.periodSeconds | int | 5 | How often (in seconds) to perform the readiness probe |
| replicaCount | int | 1 | Number of replicas |
| resources | object | {"limits":{"cpu":4,"memory":"16Gi","nvidia.com/gpu":1},"requests":{"cpu":4,"memory":"16Gi","nvidia.com/gpu":1}} | Resource configuration |
| resources.limits."nvidia.com/gpu" | int | 1 | Number of GPUs used |
| resources.limits.cpu | int | 4 | Number of CPUs |
| resources.limits.memory | string | "16Gi" | CPU memory configuration |
| resources.requests."nvidia.com/gpu" | int | 1 | Number of GPUs used |
| resources.requests.cpu | int | 4 | Number of CPUs |
| resources.requests.memory | string | "16Gi" | CPU memory configuration |
| secrets | object | {} | Secrets configuration |
| serviceName | string | "" | Service name |
| servicePort | int | 80 | Service port |
| labels.environment | string | test | Environment name |

## Configuration Examples

### Using S3 Model Download (Default)

```yaml
extraInit:
  modelDownload:
    enabled: true
  pvcStorage: "10Gi"
  s3modelpath: "models/llama-7b"
```

### Using Custom Init Containers Only

For use cases like llm-d where you need custom sidecars without model download:

```yaml
extraInit:
  modelDownload:
    enabled: false
  initContainers:
    - name: llm-d-routing-proxy
      image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.2.0
      imagePullPolicy: IfNotPresent
      ports:
        - containerPort: 8080
          name: proxy
      securityContext:
        runAsUser: 1000
      restartPolicy: Always
  pvcStorage: "10Gi"
```
