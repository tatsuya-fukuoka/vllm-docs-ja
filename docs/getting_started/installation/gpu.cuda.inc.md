<!-- markdownlint-disable MD041 MD051 -->
--8<-- [start:installation]

vLLM には、あらかじめコンパイルされた C++ と CUDA (12.9) のバイナリが含まれています。

--8<-- [end:installation]
--8<-- [start:requirements]

- GPU: compute capability 7.5 以上（T4、RTX20xx、A100、L4、H100、B200 など）

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

!!! note
    `conda` でインストールした PyTorch は `NCCL` ライブラリを静的リンクするため、vLLM が `NCCL` を使おうとしたときに問題が起きることがあります。詳細は <https://github.com/vllm-project/vllm/issues/8420> を参照してください。

vLLM は性能を出すために多数の CUDA カーネルをコンパイルする必要があります。残念ながらこのコンパイルは、他の CUDA バージョンや PyTorch バージョンとのバイナリ非互換をもたらします。同じ PyTorch バージョンでもビルド設定が異なれば互換性がありません。

そのため、vLLM は**まっさらな新しい**環境にインストールすることを推奨します。CUDA のバージョンが異なる場合や、既存の PyTorch を使いたい場合は、ソースから vLLM をビルドする必要があります。詳細は[以下](#build-wheel-from-source)を参照してください。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

```bash
uv pip install vllm --torch-backend=auto
```

??? console "pip"
    ```bash
    # Install vLLM with CUDA 12.9.
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
    ```

`uv` の `--torch-backend=auto`（または `UV_TORCH_BACKEND=auto`）を使うと、インストール済みの CUDA ドライバのバージョンを調べて[実行時に適切な PyTorch のインデックスを自動選択](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)できるため、これを推奨します。特定のバックエンド（例: `cu130`）を選ぶには `--torch-backend=cu130`（または `UV_TORCH_BACKEND=cu130`）を指定します。うまくいかない場合は、まず `uv self update` で `uv` を更新してみてください。

!!! note
    NVIDIA Blackwell 世代の GPU（B200、GB200）は CUDA 12.8 以上が必要です。その版以降の PyTorch の wheel をインストールしてください。PyTorch 側でも、対象の構成に応じた適切な pip コマンドを調べられる[専用のページ](https://pytorch.org/get-started/locally/)が用意されています。

現時点では、vLLM のバイナリは既定で CUDA 12.9 と公開版の PyTorch でコンパイルされています。CUDA 12.8 および 13.0 と公開版 PyTorch でコンパイルしたバイナリも提供しています。

```bash
# Install vLLM with a specific CUDA version (e.g., 13.0).
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # or other
export CPU_ARCH=$(uname -m) # x86_64 or aarch64
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_28_${CPU_ARCH}.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

#### 最新のコードをインストールする { #install-the-latest-code }

LLM 推論は変化の速い分野で、最新のコードにはまだリリースされていないバグ修正・性能改善・新機能が含まれていることがあります。次のリリースを待たずに最新のコードを試せるよう、vLLM は `v0.5.3` 以降のすべてのコミットについて <https://wheels.vllm.ai/nightly> で wheel を提供しています。利用できるインデックスは複数あります。

- `https://wheels.vllm.ai/nightly`: 既定のバリアント（`VLLM_MAIN_CUDA_VERSION` で指定された CUDA バージョン）を `main` ブランチの最新コミットでビルドしたもの。現時点では CUDA 12.9 です。
- `https://wheels.vllm.ai/nightly/<variant>`: その他すべてのバリアント。現在は `cu130` と `cpu` を含みます。一貫性のため、既定のバリアント（`cu129`）にもサブディレクトリがあります。

nightly のインデックスからインストールするには次を実行します。

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly # add variant subdirectory here if needed
```

!!! warning "`pip` に関する注意"

    nightly のインデックスからのインストールに `pip` を使うことは*サポートされていません*。`pip` は `--extra-index-url` と既定のインデックスのパッケージをまとめて扱い、最新バージョンのみを選ぶため、リリース版より前の開発版をインストールしづらいためです。一方 `uv` は追加のインデックスを[既定のインデックスより優先](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes)します。

    どうしても `pip` を使う場合は、wheel ファイルの完全な URL（Web ページから取得できます）を指定する必要があります。

    ```bash
    pip install -U https://wheels.vllm.ai/2f3f441f84bd5b35ec8aa9fcfffb540f107da8a7/vllm-0.23.1rc1.dev901%2Bg2f3f441f8-cp38-abi3-manylinux_2_28_x86_64.whl # current nightly build (the filename will change!)
    pip install -U https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-0.23.1rc1.dev901%2Bg2f3f441f8-cp38-abi3-manylinux_2_28_x86_64.whl # from specific commit
    ```

##### 特定のリビジョンをインストールする { #install-specific-revisions }

過去のコミットの wheel を使いたい場合（挙動の変化や性能退行の二分探索など）は、URL にコミットハッシュを指定できます。

```bash
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # use full commit hash from the main branch
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} # add variant subdirectory here if needed
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

#### Python のみのビルド（コンパイルなし）でセットアップする {#python-only-build}

Python のコードだけを変更する場合は、コンパイルなしで vLLM をビルド・インストールできます。`uv pip` の [`--editable` フラグ](https://docs.astral.sh/uv/pip/packages/#editable-packages)を使うと、変更内容が vLLM の実行に反映されます。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

このコマンドは次の処理を行います。

1. vLLM のクローンで現在のブランチを調べます。
1. main ブランチ上の対応するベースコミットを特定します。
1. そのベースコミットのビルド済み wheel をダウンロードします。
1. その中のコンパイル済みライブラリと `vllm-rs` バイナリをインストールに使います。

!!! note
    1. C++ やカーネルのコードを変更する場合、Python のみのビルドは使えません。使うと、ライブラリが見つからない、あるいは未定義シンボルといった import エラーになります。
    2. 開発ブランチをリベースした場合は、vllm をアンインストールして上記のコマンドを再実行し、ライブラリを最新に保つことを推奨します。

!!! tip "Rust フロントエンドの再ビルド"
`vllm-rs` の Rust フロントエンドのバイナリを再コンパイルしたい場合は、pip install をやり直さずに再ビルドしてインストールできます。

    ```bash
    ./build_rust.sh          # release build
    ./build_rust.sh --debug  # faster build for development
    ```

    必要であれば Rust のツールチェーンをインストールし、バイナリをビルドして `vllm/vllm-rs` に配置します。

上記のコマンドで wheel が見つからないというエラーが出る場合、ベースにした `main` のコミットがマージされたばかりで、ビルド済み wheel がまだ用意されていない可能性があります。1 時間ほど待って再試行するか、`VLLM_PRECOMPILED_WHEEL_COMMIT=nightly` を設定して、`main` 上でビルド済みの最新コミットを自動選択してください。

```bash
export VLLM_PRECOMPILED_WHEEL_COMMIT=nightly
export VLLM_USE_PRECOMPILED=1
uv pip install --editable .
```

Python のみのビルドの挙動を制御する環境変数は他にもあります。

- `VLLM_PRECOMPILED_WHEEL_LOCATION`: 使用するビルド済み wheel の URL またはローカルのファイルパスを直接指定します。wheel を探す他のロジックはすべてスキップされます。
- `VLLM_PRECOMPILED_WHEEL_COMMIT`: ダウンロードするビルド済み wheel のコミットハッシュを上書きします。`nightly` を指定すると、main ブランチ上で**すでにビルド済み**の最新コミットを使います。
- `VLLM_PRECOMPILED_WHEEL_VARIANT`: nightly のインデックスで使うバリアントのサブディレクトリ（`cu129`、`cu130`、`cpu` など）を指定します。指定しない場合は、システムの CUDA バージョン（PyTorch または nvidia-smi から取得）にもとづいて自動判定されます。`VLLM_MAIN_CUDA_VERSION` を設定して自動判定を上書きすることもできます。

vLLM の wheel についての詳細は[最新のコードをインストールする](#install-the-latest-code)を参照してください。

!!! note
    手元のソースコードのコミット ID が最新の vLLM wheel と異なる可能性があり、原因不明のエラーにつながることがあります。
    ソースコードとインストール済みの vLLM wheel は同じコミット ID を使うことを推奨します。指定した wheel のインストール方法は[最新のコードをインストールする](#install-the-latest-code)を参照してください。

#### フルビルド（コンパイルあり） {#full-build}

!!! note "コンパイラの要件"
    ソースからのビルドには GCC/G++ 11.3 以上が必要です。PyTorch の C++20 ヘッダーは
    GCC 10 や 11.3 未満の GCC とは互換性がありません。Ubuntu 22.04 の場合:
    ```bash
    sudo apt-get install -y gcc-11 g++-11
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11
    ```

C++ や CUDA のコードを変更したい場合は、ソースから vLLM をビルドする必要があります。数分かかることがあります。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -e . --torch-backend=auto
```

!!! tip
    ソースからのビルドは大量のコンパイルを伴います。繰り返しビルドする場合は、コンパイル結果をキャッシュすると効率的です。

    たとえば `conda install ccache` や `apt install ccache` で [ccache](https://github.com/ccache/ccache) をインストールできます。
    `which ccache` で `ccache` のバイナリが見つかる状態であれば、ビルドシステムが自動的に利用します。初回のビルド以降は大幅に高速になります。

    `pip install -e .` で `ccache` を使う場合は、`CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .` を実行してください。`pip` はビルドごとにランダムな名前のフォルダを作るため、`ccache` が同じファイルのビルドだと認識できなくなるためです。

    [sccache](https://github.com/mozilla/sccache) は `ccache` と同様に動作しますが、リモートストレージ上のキャッシュも利用できます。
    vLLM の `sccache` のリモート設定には次の環境変数を使えます: `SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`。あわせて `SCCACHE_IDLE_TIMEOUT=0` の設定も推奨します。

!!! note "カーネル開発を速くする"
    C++ / CUDA のカーネルを頻繁に変更する場合は、最初の `uv pip install -e .` の後に[インクリメンタルコンパイルのワークフロー](../../contributing/incremental_build.md)を使うと、変更したカーネルのみを大幅に速く再ビルドできます。

##### 既存の PyTorch を使う { #use-an-existing-pytorch-installation }

PyTorch の依存関係を `uv` で簡単にインストールできない場面があります。たとえば、既定でない PyTorch のビルド（nightly や独自ビルド）で vLLM をビルドする場合です。

既存の PyTorch を使って vLLM をビルドするには次のようにします。

```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
uv pip install --no-build-isolation -e .
```

あるいは、仮想環境の作成と管理に `uv` だけを使っている場合は、特定のパッケージについてビルドの分離を無効にする[独自の仕組み](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation)があります。
vLLM はこの仕組みを利用して、`torch` をビルド分離の対象外に指定できます。

```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
# pip install -e . does not work directly, only uv can do this
uv pip install -e .
```

##### ローカルの cutlass を使ってコンパイルする { #use-the-local-cutlass-for-compilation }

現在、vLLM はビルドを開始する前に GitHub から cutlass のコードを取得します。ただし、ローカルにある cutlass を使いたい場合もあります。
その場合は、環境変数 VLLM_CUTLASS_SRC_DIR にローカルの cutlass のディレクトリを指定してください。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass uv pip install -e . --torch-backend=auto
```

##### トラブルシューティング { #troubleshooting }

システムの負荷を抑えるため、同時に実行するコンパイルジョブの数を
環境変数 `MAX_JOBS` で制限できます。例:

```bash
export MAX_JOBS=6
uv pip install -e .
```

これは非力なマシンでビルドする場合に特に有効です。たとえば WSL では[既定で全メモリの 50% しか割り当てられない](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings)ため、`export MAX_JOBS=1` にすると複数ファイルの同時コンパイルによるメモリ不足を避けられます。
副作用として、ビルドはかなり遅くなります。

また、vLLM のビルドがうまくいかない場合は、NVIDIA の PyTorch Docker イメージの利用を推奨します。

```bash
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

Docker を使いたくない場合は、CUDA Toolkit を完全にインストールすることを推奨します。[公式サイト](https://developer.nvidia.com/cuda-toolkit-archive)からダウンロードしてインストールできます。インストール後は、環境変数 `CUDA_HOME` に CUDA Toolkit のインストール先を設定し、`nvcc` コンパイラが `PATH` に含まれるようにしてください。例:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

CUDA Toolkit が正しくインストールされているかを確認する簡単なチェックです。

```bash
nvcc --version # verify that nvcc is in your PATH
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME
```

#### サポート外の OS でのビルド { #unsupported-os-build }

vLLM が完全に動作するのは Linux だけですが、開発目的であれば他のシステム（macOS など）でもビルドでき、import や開発環境の利便性を得られます。バイナリはコンパイルされず、Linux 以外では動作しません。

インストール前に環境変数 `VLLM_TARGET_DEVICE` を無効にするだけです。

```bash
export VLLM_TARGET_DEVICE=empty
uv pip install -e .
```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

vLLM はデプロイ用に公式の Docker イメージを提供しています。
このイメージは OpenAI 互換サーバーの実行に使え、Docker Hub の [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags) で公開されています。

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

このイメージは [Podman](https://podman.io/) など他のコンテナエンジンでも利用できます。

```bash
podman run --device nvidia.com/gpu=all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=$HF_TOKEN" \
-p 8000:8000 \
--ipc=host \
docker.io/vllm/vllm-openai:latest \
--model Qwen/Qwen3-0.6B
```

イメージタグ（`vllm/vllm-openai:latest`）の後ろに、必要な[エンジン引数](https://docs.vllm.ai/en/latest/configuration/engine_args/)を追加できます。

!!! note
    コンテナがホストの共有メモリにアクセスできるよう、`ipc=host` フラグまたは `--shm-size` フラグを指定できます。
    vLLM は PyTorch を使っており、PyTorch は内部でプロセス間のデータ共有に共有メモリを使います。
    特にテンソル並列の推論で必要になります。

!!! note
    ライセンス上の問題を避けるため、任意の依存パッケージは含まれていません（例: <https://github.com/vllm-project/vllm/issues/8030>）。

    それらの依存パッケージが必要な場合は（ライセンス条項に同意したうえで）、
    ベースイメージの上にインストール用のレイヤーを追加した独自の Dockerfile を作成してください。

    ```Dockerfile
    FROM vllm/vllm-openai:v0.11.0

    # e.g. install the `audio` optional dependencies
    # NOTE: Make sure the version of vLLM matches the base image!
    RUN uv pip install --system vllm[audio]==0.11.0
    ```

!!! tip
    新しいモデルの中には、[HF Transformers](https://github.com/huggingface/transformers) の main ブランチにしか入っていないものがあります。

    開発版の `transformers` を使うには、ベースイメージの上に
    ソースからインストールするレイヤーを追加した独自の Dockerfile を作成してください。

    ```Dockerfile
    FROM vllm/vllm-openai:latest

    RUN uv pip install --system git+https://github.com/huggingface/transformers.git
    ```

#### 古い CUDA ドライバのシステムで実行する { #running-on-systems-with-older-cuda-drivers }

vLLM の Docker イメージには [CUDA 互換ライブラリ](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)があらかじめインストールされています。これにより、イメージのビルドに使われた CUDA Toolkit より古い NVIDIA ドライバのシステムでも vLLM を実行できます。

この機能を有効にするには、コンテナの実行時に環境変数 `VLLM_ENABLE_CUDA_COMPATIBILITY` を `1` または `true` に設定します。

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    --env "VLLM_ENABLE_CUDA_COMPATIBILITY=1" \
    vllm/vllm-openai <args...>
```

これにより、PyTorch などの依存パッケージを読み込む前に `LD_LIBRARY_PATH` が互換ライブラリを指すよう自動的に設定されます。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

付属の [docker/Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) を使って、ソースから vLLM をビルドして実行できます。ビルドするには次のようにします。

```bash
# optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

!!! note
    既定では、vLLM は幅広く配布できるようすべての GPU 種別向けにビルドします。実行中のマシンの GPU 種別だけを対象にビルドする場合は、
    `--build-arg torch_cuda_arch_list=""` を指定すると、vLLM が現在の GPU 種別を判定してそれ向けにビルドします。


    Docker ではなく Podman を使う場合、[既知の問題](https://github.com/containers/buildah/discussions/4184)を避けるため、
    `podman build` の実行時に `--security-opt label=disable` を付けて SELinux のラベル付けを無効にする必要があるかもしれません。

!!! note
    C++ や CUDA のカーネルコードを変更していない場合は、ビルド済み wheel を使って Docker のビルド時間を大幅に短縮できます。

    *   **有効化**: ビルド引数 `--build-arg VLLM_USE_PRECOMPILED="1"` を追加します。
    *   **仕組み**: 既定では、上流 `main` ブランチとのマージベースのコミットを使って、[Nightly Builds](https://docs.vllm.ai/en/latest/contributing/ci/nightly_builds/) から適切な wheel を自動的に探します。
    *   **コミットの上書き**: 特定のコミットの wheel を使うには `--build-arg VLLM_PRECOMPILED_WHEEL_COMMIT=<commit_hash>` を指定します。

    詳しい説明は [Build wheel from source](https://docs.vllm.ai/en/latest/contributing/ci/nightly_builds/#precompiled-wheels-usage) の「Python のみのビルド（コンパイルなし）でセットアップする」の項を参照してください。これらの引数は同じ仕組みを利用しています。

#### Arm64/aarch64 向けに vLLM の Docker イメージをソースからビルドする { #building-vllms-docker-image-from-source-for-arm64aarch64 }

Nvidia Grace-Hopper や Grace-Blackwell のような aarch64 のシステム向けに Docker コンテナをビルドできます。`--platform "linux/arm64"` フラグを付けると arm64 向けにビルドされます。

!!! note
    多数のモジュールをコンパイルするため、この処理には時間がかかります。ビルドを高速化するには `--build-arg max_jobs=` と `--build-arg nvcc_threads=` の
    フラグの利用を推奨します。ただし効果を最大化するには、`max_jobs` を `nvcc_threads` より十分大きくしてください。
    並列ジョブのメモリ使用量はかなり大きくなることがあるため注意してください（以下の例を参照）。

??? console "Command"

    ```bash
    # Example of building on Nvidia GH200 server. (Memory usage: ~15GB, Build time: ~1475s / ~25 min, Image size: 6.93GB)
    DOCKER_BUILDKIT=1 docker build . \
    --file docker/Dockerfile \
    --target vllm-openai \
    --platform "linux/arm64" \
    -t vllm/vllm-gh200-openai:latest \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --build-arg RUN_WHEEL_CHECK=false
    ```

(G)B300 では、次のコマンドのように CUDA 13 の使用を推奨します。

??? console "Command"

    ```bash
    DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA_VERSION=13.0.2 \
    --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu22.04 \
    --build-arg max_jobs=256 \
    --build-arg nvcc_threads=2 \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg torch_cuda_arch_list='9.0 10.0+PTX' \
    --platform "linux/arm64" \
    --tag vllm/vllm-gb300-openai:latest \
    --target vllm-openai \
    -f docker/Dockerfile \
    .
    ```

!!! note
    ARM 以外のホスト（x86_64 のマシンなど）で `linux/arm64` のイメージをビルドする場合は、QEMU によるクロスコンパイルの準備が必要です。これにより、ホストマシンで ARM64 の実行をエミュレートできます。

    ホストマシンで次のコマンドを実行し、QEMU の user static ハンドラを登録します。

    ```bash
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    ```

    QEMU の設定後は、`docker build` コマンドで `--platform "linux/arm64"` フラグを使えます。

#### 独自ビルドの vLLM Docker イメージを使う { #use-the-custom-built-vllm-docker-image }

独自ビルドの Docker イメージで vLLM を実行するには次のようにします。

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```

引数の `vllm/vllm-openai` は実行するイメージを指定するもので、独自ビルドしたイメージ名（ビルドコマンドの `-t` タグ）に置き換えてください。

!!! note
    **バージョン 0.4.1 と 0.4.2 のみ** - これらのバージョンの vLLM の Docker イメージは root ユーザーで実行する必要があります。実行時に root ユーザーのホームディレクトリ配下のライブラリ（`/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`）が必要になるためです。

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

機能のサポート状況は[機能 × ハードウェア](../../features/README.md#feature-x-hardware)の互換性マトリクスを参照してください。

--8<-- [end:supported-features]
