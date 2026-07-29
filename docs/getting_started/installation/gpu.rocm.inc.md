<!-- markdownlint-disable MD041 MD051 -->
--8<-- [start:installation]

vLLM は ROCm 6.3 以降の AMD GPU をサポートします。ビルド済み wheel は ROCm 7.0 と ROCm 7.2.1 向けに提供されています。

#### ビルド済み wheel

| ROCm バリアント | Python バージョン | ROCm バージョン | glibc 要件 | 対応バージョン |
| ------------ | -------------- | ------------ | ----------------- | ------------------ |
| `rocm700` | 3.12 | 7.0 | >= 2.35 | `0.14.0` 〜 `0.18.0` |
| `rocm721` | 3.12 | 7.2.1 | >= 2.35 | コミット `171775f306a333a9cf105bfd533bf3e113d401d9` 以降の nightly リリース |

--8<-- [end:installation]
--8<-- [start:requirements]

- GPU: MI200 シリーズ（gfx90a）、MI300（gfx942）、MI350（gfx950）、Radeon RX 7900 シリーズ（gfx1100/1101）、Radeon RX 9000 シリーズ（gfx1200/1201）、Ryzen AI MAX / AI 300 シリーズ（gfx1151/1150）
- ROCm 6.3 以降
    - MI350 には ROCm 7.0 以降が必要です
    - Ryzen AI MAX / AI 300 シリーズには ROCm 7.0.2 以降が必要です

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

vLLM の wheel には PyTorch と必要な依存関係がすべて同梱されており、互換性のためには同梱の PyTorch を使うべきです。vLLM は検証済みで高性能なスタックを実現するために多数の ROCm カーネルをコンパイルするため、生成されるバイナリは他の ROCm や PyTorch のビルドと互換でない場合があります。
別の ROCm バージョンが必要な場合や、既存の PyTorch インストールを使いたい場合は、vLLM をソースからビルドする必要があります。詳細は[以下](#build-wheel-from-source)を参照してください。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

!!! warning "ROCm 版 wheel には Python 3.12 が必要です"

    ROCm のビルド済み wheel は **Python 3.12** 向けのみ提供されています。別の Python バージョン（3.11 や 3.13 など）を使っている場合、インストーラは**警告なく** PyPI の CUDA 版 wheel にフォールバックし、AMD GPU では `libcudart.so: cannot open shared object file` のようなエラーで失敗します。

    Python のバージョンは `python3 --version` で確認できます。

    Python 3.12 が必要な場合は、`uv` で隔離された環境を作成できます。

    ```bash
    uv venv --python 3.12 --seed --managed-python
    source .venv/bin/activate
    ```

Python 3.12、ROCm 7.0、`glibc >= 2.35` の環境に最新版の vLLM をインストールするには次のようにします。

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
```

!!! tip
    最新の vLLM がどの ROCm バージョンをサポートしているかは、extra-index-url <https://wheels.vllm.ai/rocm/> のインデックスにある `vllm` パッケージ（[https://wheels.vllm.ai/rocm/vllm](https://wheels.vllm.ai/rocm/vllm)）で確認できます。

    別の方法として、次のコマンドで wheel のバリアントを自動的に抽出することもできます。

    ```bash
    # automatically extract the available rocm variant
    export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/vllm | grep -oP 'rocm\d+' | head -1)

    # automatically extract the vLLM version
    export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/vllm | grep -oP 'vllm-\K[0-9.]+' | head -1)

    # inspect if the ROCm version is compatible with your environment
    echo $VLLM_ROCM_VARIANT
    echo $VLLM_VERSION
    ```

特定のバージョンと ROCm バリアントの vLLM wheel をインストールするには次のようにします。

```bash
# version without the `v`
uv pip install vllm==${VLLM_VERSION} --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}

# Example
uv pip install vllm==0.18.0 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700
```

!!! warning "`pip` を使う場合の注意点"

    vLLM の wheel のインストールには `uv` の利用を推奨します。`pip` はカスタムインデックスからのインストールが煩雑です。`pip` は `--extra-index-url` と既定のインデックスのパッケージをまとめて扱い、最新バージョンのみを選ぶためです。そのため、すべてのパッケージの正確なバージョンを指定しない限り、カスタムインデックスから wheel をインストールするのは困難です。これに対して `uv` は、追加インデックスに[既定のインデックスより高い優先度](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)を与えます。

    どうしても `pip` を使いたい場合は、パッケージ名で vLLM のバージョンを正確に指定し、`--extra-index-url` でカスタムインデックス URL `https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}` を指定する必要があります。

    ```bash
    pip install vllm==0.18.0+rocm700 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700
    ```

#### 最新のコードをインストールする

LLM 推論は急速に進化している分野であり、最新のコードには未リリースのバグ修正・性能改善・新機能が含まれている場合があります。次のリリースを待たずに最新のコードを試せるよう、vLLM はコミット `171775f306a333a9cf105bfd533bf3e113d401d9` 以降のすべてのコミットについて、<https://wheels.vllm.ai/rocm/nightly/> で wheel を提供しています。使用するカスタムインデックスは `https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT}` です。

**注:** nightly の wheel をサポートする最初の ROCm バリアントは ROCm 7.2.1 です。

最新の nightly インデックスからインストールするには、次を実行します。

```bash
# automatically extract the available rocm variant
export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/nightly | \
    grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

# inspect if the ROCm version is compatible with your environment
echo $VLLM_ROCM_VARIANT

uv pip install --pre vllm \
    --extra-index-url https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT} \
    --index-strategy unsafe-best-match
```

##### 特定のリビジョンをインストールする

過去のコミットの wheel を利用したい場合（挙動の変化や性能リグレッションを二分探索するときなど）は、URL にコミットハッシュを指定できます。例:

```bash
export VLLM_COMMIT=5b8c30d62b754b575e043ce2fc0dcbf8a64f6306

export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
    grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

# Extract the version from the wheel URL
export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
    grep -oP 'vllm-\K[^-]+' | head -1  | sed 's/%2B/+/g')

# inspect the version if it is compatible with the ROCm version of your environment
echo $VLLM_ROCM_VARIANT
echo $VLLM_VERSION

uv pip install vllm==${VLLM_VERSION} \
  --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT} \
  --index-strategy unsafe-best-match
```

!!! warning "`pip` に関する注意点"

    nightly インデックスからのインストールに `pip` を使うことは_サポートされていません_。`pip` は `--extra-index-url` と既定のインデックスのパッケージをまとめて扱い、最新バージョンのみを選ぶため、リリース版より前の開発版をインストールするのが困難だからです。これに対して `uv` は、追加インデックスに[既定のインデックスより高い優先度](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)を与えます。

    どうしても `pip` を使いたい場合は、パッケージ名で vLLM のバージョンを正確に指定し、カスタムインデックス URL（Web ページから取得できます）を指定する必要があります。

    ```bash
    export VLLM_COMMIT=5b8c30d62b754b575e043ce2fc0dcbf8a64f6306

    export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
        grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

    # Extract the version from the wheel URL
    export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
        grep -oP 'vllm-\K[^-]+' | head -1  | sed 's/%2B/+/g')

    # inspect the version if it is compatible with the ROCm version of your environment
    echo $VLLM_ROCM_VARIANT
    echo $VLLM_VERSION

    pip install vllm==${VLLM_VERSION} \
    --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}
    ```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

!!! tip
    - 以下のインストール手順がうまくいかない場合は、[docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) を参照してください。Dockerfile もインストール手順の一形態です。

0. 前提条件をインストールします（以下がインストール済みの環境 / Docker をすでに使っている場合はスキップしてください）。

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    PyTorch のインストールは、`rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0` や `rocm/pytorch-nightly` などのまっさらな Docker イメージから始められます。Docker イメージを使う場合は、ステップ 3 まで飛ばして構いません。

    あるいは、PyTorch の wheel を使ってインストールすることもできます。手順は PyTorch の [Getting Started](https://pytorch.org/get-started/locally/) を参照してください。例:

    ```bash
    # Install PyTorch
    pip uninstall torch -y
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
    ```

1. [ROCm 向け Triton](https://github.com/ROCm/triton.git) をインストールします。

    [ROCm/triton](https://github.com/ROCm/triton.git) の手順に従って ROCm 版 Triton をインストールします。

    ```bash
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton.git
    cd triton
    # git checkout $TRITON_BRANCH
    git checkout f9e5bf54
    if [ ! -f setup.py ]; then cd python; fi
    python3 setup.py install
    cd ../..
    ```

    !!! note
        - 検証済みの `$TRITON_BRANCH` は [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) で確認できます。
        - triton のビルド中にパッケージのダウンロードに関する HTTP エラーが出た場合は、一時的なものなので再試行してください。

2. 任意: CK flash attention を使う場合は、[ROCm 向け flash attention](https://github.com/Dao-AILab/flash-attention.git) をインストールできます。

    [ROCm/flash-attention](https://github.com/Dao-AILab/flash-attention#amd-rocm-support) の手順に従って ROCm 版 flash attention（v2.8.0）をインストールします。

    たとえば ROCm 7.0 で、gfx アーキテクチャが `gfx942` の場合は次のようにします。gfx アーキテクチャは `rocminfo |grep gfx` で確認できます。

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    # git checkout $FA_BRANCH
    git checkout 0e60e394
    git submodule update --init
    GPU_ARCHS="gfx942" python3 setup.py install
    cd ..
    ```

    !!! note
        - 検証済みの `$FA_BRANCH` は [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) で確認できます。

3. 任意: 特定のブランチやコミットを使うために AITER を自分でビルドする場合は、次の手順でビルドできます。

    ```bash
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    git checkout $AITER_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        - 目的に応じて `$AITER_BRANCH_OR_COMMIT` を設定する必要があります。
        - 検証済みの `$AITER_BRANCH_OR_COMMIT` は [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) で確認できます。

4. 任意: EP や PD 分離のために MORI を使いたい場合は、次の手順で [MORI](https://github.com/ROCm/mori) をインストールできます。

    ```bash
    git clone https://github.com/ROCm/mori.git
    cd mori
    git checkout $MORI_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    MORI_GPU_ARCHS="gfx942;gfx950" python3 setup.py install
    ```

    !!! note
        - 目的に応じて `$MORI_BRANCH_OR_COMMIT` を設定する必要があります。
        - 検証済みの `$MORI_BRANCH_OR_COMMIT` は [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) で確認できます。

5. vLLM をビルドします。たとえば ROCm 7.0 上の vLLM は次の手順でビルドできます。

    ???+ console "コマンド"

        ```bash
        pip install --upgrade pip

        # Build & install AMD SMI
        pip install /opt/rocm/share/amd_smi

        # Install dependencies
        pip install --upgrade numba \
            scipy \
            huggingface-hub[cli] \
            setuptools_scm
        pip install -r requirements/rocm.txt

        # To build for a single architecture (e.g., MI300) for faster installation (recommended):
        export PYTORCH_ROCM_ARCH="gfx942"

        # To build vLLM for multiple arch MI210/MI250/MI300, use this instead
        # export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

        python3 setup.py develop
        ```

    これには 5〜10 分ほどかかることがあります。現時点では、ROCm 環境でソースから vLLM をインストールする際に `pip install .` は動作しません。

    !!! tip
        - PyTorch の ROCm バージョンは、理想的には ROCm ドライバのバージョンと一致させるべきです。

!!! tip
    - MI300x（gfx942）を使う場合、最適な性能を得るために、システムやワークフローレベルの性能最適化とチューニングのヒントについては [MI300x チューニングガイド](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)を参照してください。
      vLLM については [vLLM の性能最適化](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)を参照してください。

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

vLLM はデプロイ向けの公式 Docker イメージを提供しています。
これらのイメージは OpenAI 互換サーバーの実行に使え、Docker Hub の [vllm/vllm-openai-rocm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags) から入手できます。

- `vllm/vllm-openai-rocm:latest` — 安定版リリース
- `vllm/vllm-openai-rocm:nightly` — 最新の開発ブランチからのプレビュービルド。最新の機能や修正が必要な場合はこちらを使ってください

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm:<tag> \
    --model Qwen/Qwen3-0.6B
```

開発のベースとしてこの Docker イメージを使う場合、エントリポイントを上書きして対話セッションで起動できます。

???+ console "コマンド"
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint /bin/bash \
        vllm/vllm-openai-rocm:<tag>
    ```

#### AMD の Docker イメージを使う（非推奨）

!!! warning "非推奨"
    AMD の Docker イメージ（`rocm/vllm` および `rocm/vllm-dev`）は、上記の vLLM 公式 Docker イメージ（`vllm/vllm-openai-rocm`）に置き換えられ非推奨となりました。公式イメージへ移行してください。

公式 Docker イメージが[上流の vLLM Docker Hub](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/) で提供されるようになった 2026 年 1 月 20 日より前は、[AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) が、AMD Instinct MI300X™ アクセラレータ上での推論性能の検証を目的とした、ビルド済みで最適化された Docker イメージを提供していました。
AMD は [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) で nightly のビルド済み Docker イメージも提供しており、vLLM とその依存関係がすべてインストールされています。この Docker イメージのエントリポイントは `/bin/bash` です（vLLM の公式 Docker イメージとは異なります）。

!!! tip
    このビルド済み Docker イメージの使い方は、[AMD Instinct MI300X における LLM 推論性能の検証](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)を参照してください。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

同梱の [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) を使って、ソースから vLLM をビルドして実行できます。

??? info "（任意）ROCm ソフトウェアスタック入りのイメージをビルドする"

    vLLM が必要とする ROCm ソフトウェアスタックをセットアップする [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) から Docker イメージをビルドします。
    **この rocm_base イメージは通常ビルド済みで、利用体験を高速化するため [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) にタグ `rocm/vllm-dev:base` として置かれているため、このステップは任意です。**
    自分で rocm_base イメージをビルドする場合の手順は次のとおりです。

    Docker のビルドは buildkit で開始することが重要です。docker build コマンドの実行時に環境変数 `DOCKER_BUILDKIT=1` を指定するか、Docker デーモンの設定 `/etc/docker/daemon.json` で次のように buildkit を有効にしてデーモンを再起動してください。

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

    MI200 / MI300 シリーズ向けに ROCm 7.0 上で vLLM をビルドするには、既定の設定を使えます。

    ```bash
    DOCKER_BUILDKIT=1 docker build \
        -f docker/Dockerfile.rocm_base \
        -t rocm/vllm-dev:base .
    ```

まず、[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) から Docker イメージをビルドし、そのイメージからコンテナを起動します。
Docker のビルドは buildkit で開始することが重要です。docker build コマンドの実行時に環境変数 `DOCKER_BUILDKIT=1` を指定するか、Docker デーモンの設定 /etc/docker/daemon.json で次のように buildkit を有効にしてデーモンを再起動してください。

```json
{
    "features": {
        "buildkit": true
    }
}
```

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) は既定で ROCm 7.0 を使いますが、古い vLLM のブランチでは ROCm 5.7、6.0、6.1、6.2、6.3、6.4 もサポートします。
次の引数で Docker イメージのビルドを柔軟にカスタマイズできます。

- `BASE_IMAGE`: `docker build` 実行時に使うベースイメージを指定します。既定値の `rocm/vllm-dev:base` は AMD が公開・保守しているイメージで、[docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) を使ってビルドされています。
- `ARG_PYTORCH_ROCM_ARCH`: ベースの Docker イメージの gfx アーキテクチャの値を上書きできます。

これらの値は、`docker build` 実行時に `--build-arg` オプションで渡せます。

MI200 / MI300 シリーズ向けに ROCm 7.0 上で vLLM をビルドするには、既定の設定を使えます（`vllm serve` をエントリポイントとする Docker イメージがビルドされます）。

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vllm-openai-rocm .
```

独自にビルドした Docker イメージで vLLM を実行するには次のようにします。

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm <args...>
```

引数 `vllm/vllm-openai-rocm` は実行するイメージを指定するもので、独自にビルドしたイメージ名（ビルドコマンドの `-t` タグ）に置き換えてください。

開発のベースとしてこの Docker イメージを使う場合、エントリポイントを上書きして対話セッションで起動できます。

???+ console "コマンド"
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint bash \
        vllm/vllm-openai-rocm
    ```

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

機能のサポート状況については、[機能 × ハードウェア](../../features/README.md#feature-x-hardware)の互換性マトリクスを参照してください。

--8<-- [end:supported-features]
