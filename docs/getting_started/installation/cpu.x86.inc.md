<!-- markdownlint-disable MD041 MD051 -->
--8<-- [start:installation]

vLLM は x86 CPU プラットフォーム上での基本的なモデル推論とサービングに対応しており、データ型は FP32、FP16、BF16 をサポートします。

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: Linux
- CPU フラグ: `avx512f`（推奨）、`avx2`（機能が制限されます）

!!! tip
    CPU フラグの確認には `lscpu` を使います。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

AVX512 / AVX2 対応の x86 向けビルド済み vLLM wheel は、バージョン 0.17.0 以降で提供されています。リリース版の wheel をインストールするには次のようにします。

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

# use uv
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl --torch-backend cpu
```

??? console "pip"
    ```bash
    # use pip
    pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cpu
    ```
!!! warning "`LD_PRELOAD` を設定する"
    wheel からインストールした vLLM CPU を使う前に、TCMalloc と Intel OpenMP がインストールされ、`LD_PRELOAD` に追加されていることを確認してください。
    ```bash
    # install TCMalloc, Intel OpenMP is installed with vLLM CPU
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```

#### 最新のコードをインストールする

最新の main ブランチからビルドされた wheel をインストールするには次のようにします。

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

#### 特定のリビジョンをインストールする

過去のコミットの wheel を利用したい場合（挙動の変化や性能リグレッションを二分探索するときなど）は、URL にコミットハッシュを指定できます。

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

推奨コンパイラをインストールします。問題を避けるため、既定のコンパイラとして `gcc/g++ >= 12.3.0` を使うことを推奨します。たとえば Ubuntu 22.4 では次のように実行します。

```bash
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

vLLM プロジェクトをクローンします。

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

必要な依存関係をインストールします。

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

vLLM をビルドしてインストールします。

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

vLLM を開発したい場合は、代わりに editable モードでインストールします。

```bash
VLLM_TARGET_DEVICE=cpu python3 setup.py develop
```

必要に応じて、別の環境でもインストールできるポータブルな wheel をビルドできます。

```bash
VLLM_TARGET_DEVICE=cpu uv build --wheel --no-build-isolation
```

```bash
uv pip install dist/*.whl
```

??? console "pip"
    ```bash
    VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation
    ```

    ```bash
    pip install dist/*.whl
    ```

!!! warning "`LD_PRELOAD` を設定する"
    wheel からインストールした vLLM CPU を使う前に、TCMalloc と Intel OpenMP がインストールされ、`LD_PRELOAD` に追加されていることを確認してください。
    ```bash
    # install TCMalloc, Intel OpenMP is installed with vLLM CPU
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```

!!! example "トラブルシューティング"
    - **NumPy 2.0 以上でのエラー**: `pip install "numpy<2.0"` でダウングレードしてください。
    - **CMake が CUDA を検出してしまう**: CUDA がインストールされていても CPU ビルド時に CUDA を検出しないよう、`CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` を追加してください。
    - `AMD` で CPU 上の vLLM を動かすには、[AVX512](https://www.phoronix.com/review/amd-zen4-avx512) をサポートする第 4 世代（Zen 4 / Genoa）以降のプロセッサが必要です。
    - `Could not find a version that satisfies the requirement torch==X.Y.Z+cpu+cpu` のようなエラーが出る場合は、pip が依存関係を解決できるよう [pyproject.toml](https://github.com/vllm-project/vllm/blob/main/pyproject.toml) の更新を検討してください。
    ```toml title="pyproject.toml"
    [build-system]
    requires = [
      "cmake>=3.26.1",
      ...
      "torch==X.Y.Z+cpu"   # <-------
    ]
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

Docker Hub から最新の CPU 向けイメージを取得できます。

```bash
docker pull vllm/vllm-openai-cpu:latest-x86_64
```

特定の vLLM バージョンのイメージを取得するには次のようにします。

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64
```

利用可能なイメージタグの一覧は [https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags) にあります。

これらのイメージは次のように実行できます。

```bash
docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai-cpu:latest-x86_64 <args...>
```

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

#### 対象 CPU 向けにビルドする

```bash
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_X86=<false (default)|true> \ # For cross-compilation
        --tag vllm-cpu-env \
        --target vllm-openai .
```

#### AMD Zen 最適化を有効にしてビルドする

AMD Zen 4 / Zen 5 のホスト（`linux/amd64` のみ）では、`vllm-openai-zen` ターゲットを使います。これは既定の `vllm-openai` イメージを拡張し、`vllm[zen]` エクストラ経由で `zentorch` を追加するもので、実行時に `ZenCpuPlatform` が自動で有効になります。

```bash
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-zen-env \
        --target vllm-openai-zen .
```

生成されるイメージは `vllm-openai` と同じ引数・環境変数を受け付けます（下記の [OpenAI サーバーを起動する](#launching-the-openai-server) を参照）。Zen 最適化を有効にするための追加フラグは不要です。実行時の挙動とサポートされる dtype の注意点については [AMD Zen 最適化](cpu.md#amd-zen-optimizations) を参照してください。

#### OpenAI サーバーを起動する {#launching-the-openai-server}

```bash
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            vllm-cpu-env \
            meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

--8<-- [end:build-image-from-source]
--8<-- [start:amd-zen-optimizations]

AMD Zen CPU では、vLLM は `ZenCpuPlatform`（`CpuPlatform` のサブクラス）を自動的に選択し、線形層を [`zentorch`](https://github.com/amd/ZenDNN-pytorch-plugin) の ZenDNN 最適化カーネル経由でディスパッチします。インストールコマンドについては FAQ の [AMD Zen 最適化を有効にするには？](#how-do-i-enable-amd-zen-optimizations) を参照してください。

### 判定ルール

`ZenCpuPlatform` は、次の条件が**すべて**満たされる場合に選択されます。

- vLLM が CPU 向けにビルドされている
- `/proc/cpuinfo` が `AuthenticAMD` と `avx512` を報告する
- `import zentorch` が成功する

それ以外の場合、vLLM は既定の `CpuPlatform`（oneDNN / sgl-kernel の経路）にフォールバックします。

### サポートされる dtype

`ZenCpuPlatform` では `float16` は**サポートされません**。`ZenCpuPlatform.supported_dtypes` が公開するのは `bfloat16` と `float32` のみです。そのため `torch_dtype=float16` と宣言されたモデルは、ロード時に自動的に `bfloat16` へダウンキャストされ、`vllm/config/model.py` から `"Your device 'cpu' doesn't support torch.float16. Falling back to torch.bfloat16 for compatibility."` という標準の警告が出力されます。

### 環境変数

- `VLLM_ZENTORCH_WEIGHT_PREPACK`（既定値 `1`）: モデルのロード時に線形層の重みを ZenDNN のブロック化レイアウトへ先読みで prepack し、推論ごとのレイアウト変換のオーバーヘッドをなくします。無効にするには `0` を設定します。

### Docker

`vllm-openai-zen` の Docker ターゲット（`docker/Dockerfile.cpu` 内）は、既定の `vllm-openai` イメージを `vllm[zen]` で拡張したものです。`docker build -f docker/Dockerfile.cpu --target vllm-openai-zen .` でビルドします。完全なコマンドと実行手順は [AMD Zen 最適化を有効にしてビルドする](#building-with-amd-zen-optimizations) を参照してください。

### 参考

設計の背景については [RFC #35089: In-Tree AMD Zen CPU Backend via zentorch](https://github.com/vllm-project/vllm/issues/35089) を参照してください。

--8<-- [end:amd-zen-optimizations]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
