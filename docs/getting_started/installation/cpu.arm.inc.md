<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM は Arm CPU プラットフォーム上での基本的なモデル推論とサービングを提供しており、NEON をサポートし、データ型は FP32、FP16、BF16 に対応します。

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: Linux
- コンパイラ: `gcc/g++ >= 12.3.0`（任意、推奨）
- 命令セットアーキテクチャ（ISA）: NEON のサポートが必要

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

Arm 向けのビルド済み vLLM wheel は、バージョン 0.11.2 以降で提供されています。これらの wheel にはコンパイル済みの C++ バイナリが含まれます。

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_aarch64.whl --torch-backend cpu
```

??? console "pip"
    ```bash
    pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_aarch64.whl --extra-index-url https://download.pytorch.org/whl/cpu
    ```

!!! warning "`LD_PRELOAD` を設定する"
    wheel からインストールした vLLM CPU を使う前に、TCMalloc がインストールされ、`LD_PRELOAD` に追加されていることを確認してください。
    ```bash
    # install TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

`uv` を使う方法は vLLM `v0.6.6` 以降で利用できます。`uv` の特徴的な点は、`--extra-index-url` にあるパッケージが[既定のインデックスよりも高い優先度](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)を持つことです。最新の公開リリースが `v0.6.6.post1` の場合でも、`uv` の挙動であれば `--extra-index-url` を指定することで `v0.6.6.post1` より前のコミットをインストールできます。これに対して `pip` は `--extra-index-url` と既定のインデックスのパッケージをまとめて扱い、最新バージョンのみを選ぶため、リリース版より前の開発版をインストールするのが困難です。

#### 最新のコードをインストールする

LLM 推論は急速に進化している分野であり、最新のコードには未リリースのバグ修正・性能改善・新機能が含まれている場合があります。次のリリースを待たずに最新のコードを試せるよう、vLLM は `v0.11.2` 以降のすべてのコミットについて、動作するビルド済み Arm CPU wheel を <https://wheels.vllm.ai/nightly> で提供しています。ネイティブの CPU wheel には次のインデックスを使ってください。

- `https://wheels.vllm.ai/nightly/cpu/vllm`

nightly インデックスからインストールするには、次を実行します。

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

??? console "pip（注意点あり）"

    nightly インデックスからのインストールに `pip` を使うことは_サポートされていません_。`pip` は `--extra-index-url` と既定のインデックスのパッケージをまとめて扱い、最新バージョンのみを選ぶため、リリース版より前の開発版をインストールするのが困難だからです。これに対して `uv` は、追加インデックスに[既定のインデックスより高い優先度](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)を与えます。

    どうしても `pip` を使いたい場合は、wheel ファイルの完全な URL（リンクアドレス）を指定する必要があります（URL は https://wheels.vllm.ai/nightly/cpu/vllm から取得できます）。

    ```bash
    pip install https://wheels.vllm.ai/2f3f441f84bd5b35ec8aa9fcfffb540f107da8a7/vllm-0.23.1rc1.dev901%2Bg2f3f441f8.cpu-cp38-abi3-manylinux_2_34_aarch64.whl --extra-index-url https://download.pytorch.org/whl/cpu # current nightly build (the filename will change!)
    ```

#### 特定のリビジョンをインストールする

過去のコミットの wheel を利用したい場合（挙動の変化や性能リグレッションを二分探索するときなど）は、URL にコミットハッシュを指定できます。

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

まず、推奨コンパイラをインストールします。問題を避けるため、既定のコンパイラとして `gcc/g++ >= 12.3.0` を使うことを推奨します。たとえば Ubuntu 22.4 では次のように実行します。

```bash
sudo apt-get update  -y
sudo apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

次に、vLLM プロジェクトをクローンします。

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

続いて、必要な依存関係をインストールします。

```bash
uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
uv pip install -r requirements/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
```

??? console "pip"
    ```bash
    pip install --upgrade pip
    pip install -v -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

最後に、vLLM をビルドしてインストールします。

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

vLLM を開発したい場合は、代わりに editable モードでインストールします。

```bash
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

互換性の検証は AWS Graviton3 インスタンス上で実施されています。

!!! warning "`LD_PRELOAD` を設定する"
    wheel からインストールした vLLM CPU を使う前に、TCMalloc がインストールされ、`LD_PRELOAD` に追加されていることを確認してください。
    ```bash
    # install TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

Docker Hub から最新のイメージを取得するには次のようにします。

```bash
docker pull vllm/vllm-openai-cpu:latest-arm64
```

特定の vLLM バージョンのイメージを取得するには次のようにします。

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-arm64
```

利用可能なイメージタグの一覧は [https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags) にあります。

これらのイメージは次のように実行できます。

```bash
docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai-cpu:latest-arm64 <args...>
```

Docker イメージで最新のコードを利用することもできます。これらは本番利用を想定したものではなく、CI とテスト専用です。数日で失効します。

最新のコードにはバグが含まれる可能性があり、安定しているとは限りません。利用には注意してください。

```bash
export VLLM_COMMIT=6299628d326f429eba78736acb44e76749b281f5 # use full commit hash from the main branch
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}-arm64-cpu
```

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

#### 対象の ARM CPU 向けにビルドする

```bash
docker build -f docker/Dockerfile.cpu \
        --platform=linux/arm64 \
        --build-arg VLLM_CPU_ARM_BF16=<false (default)|true> \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

!!! note "既定では自動検出"
    既定では、ARM CPU の命令セット（BF16、NEON など）はビルドシステムの CPU フラグから自動的に検出されます。`VLLM_CPU_ARM_BF16` ビルド引数はクロスコンパイル用です。

    - `VLLM_CPU_ARM_BF16=true` - ARM BF16 サポートを強制的に有効にする（ビルドシステムの対応状況にかかわらず BF16 付きでビルドする）
    - `VLLM_CPU_ARM_BF16=false` - 自動検出に任せる（既定）

##### 例

###### 自動検出によるビルド（ネイティブ ARM）

```bash
# Building on ARM64 system - platform auto-detected
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-arm64 \
        --target vllm-openai .
```

###### BF16 対応の ARM 向けクロスコンパイル

```bash
# Building on ARM64 for newer ARM CPUs with BF16
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_ARM_BF16=true \
        --tag vllm-cpu-arm64-bf16 \
        --target vllm-openai .
```

###### x86_64 から ARM64（BF16 付き）へのクロスコンパイル

```bash
# Requires Docker buildx with ARM emulation (QEMU)
docker buildx build -f docker/Dockerfile.cpu \
        --platform=linux/arm64 \
        --build-arg VLLM_CPU_ARM_BF16=true \
        --build-arg max_jobs=4 \
        --tag vllm-cpu-arm64-bf16 \
        --target vllm-openai \
        --load .
```

!!! note "ARM BF16 の要件"
    ARM BF16 のサポートには ARMv8.6-A 以降（FEAT_BF16）が必要です。AWS Graviton3/4、AmpereOne、その他の最近の ARM プロセッサでサポートされています。

#### OpenAI サーバーを起動する

```bash
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-arm64 \
            meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

!!! tip "--privileged の代替"
    セキュリティを高めるため、`--privileged=true` の代わりに `--cap-add SYS_NICE --security-opt seccomp=unconfined` を使ってください。

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
