---
toc_depth: 2
---

# Docker を使う { #using-docker }

## ビルド済みイメージ { #pre-built-images }

--8<-- "docs/getting_started/installation/gpu.md:pre-built-images"

## root 以外のユーザーで実行する { #run-as-a-non-root-user }

CUDA 版の `vllm/vllm-openai` イメージは、後方互換性のため既定では root で実行されます。
組み込みの `vllm` ユーザー（UID 2000、GID 0）で実行することもできます。

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

root 以外で動かすコンテナにモデルやキャッシュのボリュームをマウントする場合は、
書き込み可能なパスを `/root` ではなく `/home/vllm` 以下にマウントしてください。
たとえば Hugging Face のキャッシュは `/home/vllm/.cache/huggingface` にマウントし、
マウントしたディレクトリをグループ 0 から書き込めるようにします。

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

既定で root 以外の `vllm` ユーザーを使うイメージをビルドするには、
オプトインの `vllm-openai-nonroot` ターゲットを指定します。

```bash
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

docker run --rm --gpus all \
    -p 8000:8000 \
    vllm-openai-nonroot:local \
    meta-llama/Llama-3.1-8B-Instruct
```

`vllm-openai-nonroot` ターゲットは、実行時の UID がグループ 0 に属していれば、
OpenShift 形式の任意の UID にも対応します。Kubernetes のマニフェストでは、
コンテナのセキュリティコンテキストを次のように設定し、マウントしたキャッシュや
モデルのパスをグループ 0 から書き込めるようにしてください。

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000540000
  runAsGroup: 0
  fsGroup: 0
```

グループ 0 に属さない実行時 UID は、`/home/vllm` や `/opt/uv/cache` に書き込めない
可能性があるため、サポート対象には含まれていません。

## ソースからイメージをビルドする { #build-image-from-source }

--8<-- "docs/getting_started/installation/gpu.md:build-image-from-source"
