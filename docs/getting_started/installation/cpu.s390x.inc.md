<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM は IBM Z プラットフォーム上の s390x アーキテクチャを実験的にサポートしています。現時点では、IBM Z プラットフォーム上でネイティブに実行するにはソースからビルドする必要があります。

現在、s390x アーキテクチャ向けの CPU 実装は FP32、BF16、FP16 をサポートしています。

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: `Linux`
- SDK: Command Line Tools を含む `gcc/g++ >= 14.0.0` 以降
- 命令セットアーキテクチャ（ISA）: VXE のサポートが必要です。Z14 以降で動作します。
- ビルド時にインストールする Python パッケージ: `torchvision`、`llvmlite`、`numba`、`pyarrow`（テスト用）、`opencv-headless`

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

現時点では、IBM Z 向けの CPU のビルド済み wheel はありません。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

vLLM をビルドする前に、パッケージマネージャから次のパッケージをインストールします。RHEL 9.6 での例は次のとおりです。

```bash
dnf install -y \
    which procps findutils tar vim git gcc-toolset-14 gcc-toolset-14-binutils gcc-toolset-14-libatomic-devel zlib-devel \
    libjpeg-turbo-devel libtiff-devel libpng-devel libwebp-devel freetype-devel harfbuzz-devel \
    openssl-devel openblas openblas-devel autoconf automake libtool cmake numpy libsndfile \
    clang llvm-devel llvm-static clang-devel
```

`outlines-core` と `uvloop` の Python パッケージのインストールに必要な rust 1.80 以上をインストールします。

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env"
```

次のコマンドを実行して、ソースから vLLM をビルド・インストールします。

!!! tip
    vLLM をビルドする前に、依存パッケージ `torchvision`、`llvmlite`、`numba`、`llguidance`、`pyarrow`、`opencv-headless` をソースからビルドしてください。

```bash
    uv pip install -v \
        -r requirements/build/cpu.txt \
        -r requirements/cpu.txt \
        --torch-backend cpu \
        --index-strategy unsafe-best-match && \
    VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel && \
        uv pip install dist/*.whl
```

??? console "pip"
    ```bash
        pip install -v \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements/build/cpu.txt \
            -r requirements/cpu.txt \
        VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel && \
            pip install dist/*.whl
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

現時点では、IBM Z 向けの CPU のビルド済みイメージはありません。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.s390x \
    --tag vllm-cpu-env .

# Launch OpenAI server
docker run --rm \
    --privileged true \
    --shm-size 4g \
    -p 8000:8000 \
    -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
    -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
    vllm-cpu-env \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dtype float \
    other vLLM OpenAI server arguments
```

!!! tip
    `--privileged true` の代わりに `--cap-add SYS_NICE --security-opt seccomp=unconfined` を使うこともできます。

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
