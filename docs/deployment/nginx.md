# Nginx を使う { #using-nginx }

このドキュメントでは、複数の vLLM サービングコンテナを起動し、Nginx をサーバー間のロードバランサーとして使う方法を説明します。

## Nginx コンテナのビルド { #build-nginx-container }

このガイドでは、vLLM のプロジェクトをクローンした直後で、vllm のルートディレクトリにいることを前提とします。

```bash
export vllm_root=`pwd`
```

`Dockerfile.nginx` というファイルを作成します。

```dockerfile
FROM nginx:latest
RUN rm /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

コンテナをビルドします。

```bash
docker build . -f Dockerfile.nginx --tag nginx-lb
```

## 簡単な Nginx 設定ファイルの作成 { #create-simple-nginx-config-file }

`nginx_conf/nginx.conf` というファイルを作成します。サーバーは好きなだけ追加できます。以下の例ではまず 2 台から始めます。追加するには、`upstream backend` に `server vllmN:8000 max_fails=3 fail_timeout=10000s;` の行を足してください。

??? console "設定"

    ```console
    upstream backend {
        least_conn;
        server vllm0:8000 max_fails=3 fail_timeout=10000s;
        server vllm1:8000 max_fails=3 fail_timeout=10000s;
    }
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
    ```

## vLLM コンテナのビルド { #build-vllm-container }

```bash
cd $vllm_root
docker build -f docker/Dockerfile . --tag vllm
```

プロキシ環境下にいる場合は、次のように docker build コマンドにプロキシ設定を渡せます。

```bash
cd $vllm_root
docker build \
    -f docker/Dockerfile . \
    --tag vllm \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy
```

## Docker ネットワークの作成 { #create-docker-network }

```bash
docker network create vllm_nginx
```

## vLLM コンテナの起動 { #launch-vllm-containers }

注意点:

- HuggingFace のモデルを別の場所にキャッシュしている場合は、以下の `hf_cache_dir` を書き換えてください。
- HuggingFace のキャッシュがまだない場合は、まず `vllm0` を起動し、モデルのダウンロードとサーバーの準備が完了するのを待ってください。こうすると `vllm1` はダウンロード済みのモデルを再利用でき、再ダウンロードが不要になります。
- 以下の例は GPU バックエンドを前提としています。CPU バックエンドを使う場合は `--gpus device=ID` を削除し、docker run コマンドに `VLLM_CPU_KVCACHE_SPACE` と `VLLM_CPU_OMP_THREADS_BIND` の環境変数を追加してください。
- `Llama-2-7b-chat-hf` 以外を使う場合は、vLLM サーバーで使うモデル名を書き換えてください。

??? console "コマンド"

    ```console
    mkdir -p ~/.cache/huggingface/hub/
    hf_cache_dir=~/.cache/huggingface/
    docker run \
        -itd \
        --ipc host \
        --network vllm_nginx \
        --gpus device=0 \
        --shm-size=10.24gb \
        -v $hf_cache_dir:/root/.cache/huggingface/ \
        -p 8081:8000 \
        --name vllm0 vllm \
        --model meta-llama/Llama-2-7b-chat-hf
    docker run \
        -itd \
        --ipc host \
        --network vllm_nginx \
        --gpus device=1 \
        --shm-size=10.24gb \
        -v $hf_cache_dir:/root/.cache/huggingface/ \
        -p 8082:8000 \
        --name vllm1 vllm \
        --model meta-llama/Llama-2-7b-chat-hf
    ```

!!! note
    プロキシ環境下にいる場合は、`-e http_proxy=$http_proxy -e https_proxy=$https_proxy` で docker run コマンドにプロキシ設定を渡せます。

## Nginx の起動 { #launch-nginx }

```bash
docker run \
    -itd \
    -p 8000:80 \
    --network vllm_nginx \
    -v ./nginx_conf/:/etc/nginx/conf.d/ \
    --name nginx-lb nginx-lb:latest
```

## vLLM サーバーの準備完了を確認する { #verify-that-vllm-servers-are-ready }

```bash
docker logs vllm0 | grep Uvicorn
docker logs vllm1 | grep Uvicorn
```

どちらの出力も次のようになるはずです。

```console
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
