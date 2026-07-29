# インクリメンタルビルドのワークフロー { #incremental-compilation-workflow }

`csrc/` ディレクトリにある vLLM の C++/CUDA カーネルを開発していると、変更のたびに `uv pip install -e .` でプロジェクト全体を再コンパイルするのは時間がかかります。CMake を使ったインクリメンタルコンパイルのワークフローなら、最初のセットアップ以降は必要なコンポーネントだけを再コンパイルできるため、より速く反復できます。このガイドでは、editable な Python インストールを補完するこのワークフローのセットアップ方法と使い方を説明します。

## 前提条件 { #prerequisites }

インクリメンタルビルドをセットアップする前に、次を確認してください。

1. **vLLM の editable インストール:** vLLM をソースから editable モードでインストールしておいてください。最初の editable セットアップにビルド済み wheel を使うと速く済みます。以降のカーネルの再コンパイルは CMake のワークフローが担当します。

    ```console
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
    ```

2. **CUDA Toolkit:** NVIDIA CUDA Toolkit が正しくインストールされ、`nvcc` が `PATH` から利用できることを確認してください。CMake は CUDA コードのコンパイルに `nvcc` を使います。`nvcc` は通常 `$CUDA_HOME/bin/nvcc` にあり、`which nvcc` でも確認できます。問題が起きた場合は、[CUDA Toolkit の公式インストールガイド](https://developer.nvidia.com/cuda-toolkit-archive)と vLLM の [GPU インストールドキュメント](../getting_started/installation/gpu.md#troubleshooting)のトラブルシューティングを参照してください。`CMakeUserPresets.json` の `CMAKE_CUDA_COMPILER` 変数も `nvcc` のバイナリを指している必要があります。

3. **ビルドツール:** コンパイル結果をキャッシュして再ビルドを高速化するため、`ccache` のインストールを強く推奨します（`sudo apt install ccache` や `conda install ccache` など）。また、`cmake` や `ninja` といった基本的なビルド依存パッケージもインストールしてください。これらは `requirements/build/cuda.txt` またはシステムのパッケージマネージャからインストールできます。

    ```console
    uv pip install -r requirements/build/cuda.txt --torch-backend=auto
    ```

## CMake ビルド環境のセットアップ { #setting-up-the-cmake-build-environment }

インクリメンタルビルドは CMake を通じて管理します。ビルド設定は、vLLM リポジトリのルートに置く `CMakeUserPresets.json` ファイルで指定できます。

### ヘルパースクリプトで `CMakeUserPresets.json` を生成する { #generate-cmakeuserpresetsjson-using-the-helper-script }

セットアップを簡単にするため、vLLM にはシステムの構成（CUDA のパス、Python 環境、CPU コア数など）を自動検出して `CMakeUserPresets.json` を生成するヘルパースクリプトが用意されています。

**スクリプトの実行:**

vLLM のクローンのルートに移動し、次のコマンドを実行します。

```console
python tools/generate_cmake_presets.py
```

一部のパス（`nvcc` や vLLM 開発環境の Python 実行ファイルなど）を自動的に判定できない場合、スクリプトが入力を求めます。画面の指示に従ってください。既存の `CMakeUserPresets.json` が見つかった場合、上書きの前に確認を求められます。

**既存ファイルを強制的に上書きする:**

確認なしで既存の `CMakeUserPresets.json` を上書きするには、`--force-overwrite` フラグを使います。

```console
python tools/generate_cmake_presets.py --force-overwrite
```

これは、対話的なプロンプトを避けたい自動化スクリプトや CI/CD 環境で特に便利です。

スクリプトを実行すると、vLLM リポジトリのルートに `CMakeUserPresets.json` が作成されます。

### `CMakeUserPresets.json` の例 { #example-cmakeuserpresetsjson }

次は、生成される `CMakeUserPresets.json` の例です。実際の値は、システムの構成と入力内容に応じて調整されます。

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 26,
        "patch": 1
    },
    "configurePresets": [
        {
            "name": "release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-release",
            "cacheVariables": {
                "CMAKE_CUDA_COMPILER": "/usr/local/cuda/bin/nvcc",
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CUDA_COMPILER_LAUNCHER": "ccache",
                "CMAKE_BUILD_TYPE": "Release",
                "VLLM_PYTHON_EXECUTABLE": "/home/user/venvs/vllm/bin/python",
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_CUDA_FLAGS": "",
                "NVCC_THREADS": "4",
                "CMAKE_JOB_POOLS": "compile=32"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "release",
            "configurePreset": "release",
            "jobs": 32
        }
    ]
}
```

**各設定の意味**

- `CMAKE_CUDA_COMPILER`: `nvcc` バイナリへのパス。スクリプトが自動検出を試みます。
- `CMAKE_C_COMPILER_LAUNCHER`、`CMAKE_CXX_COMPILER_LAUNCHER`、`CMAKE_CUDA_COMPILER_LAUNCHER`: これらを `ccache`（または `sccache`）に設定すると、コンパイル結果がキャッシュされ再ビルドが大幅に高速化します。`ccache` がインストールされていることを確認してください（`sudo apt install ccache` や `conda install ccache` など）。スクリプトは既定でこれらを設定します。
- `VLLM_PYTHON_EXECUTABLE`: vLLM 開発環境の Python 実行ファイルへのパス。スクリプトが入力を求め、適切であれば現在の Python 環境が既定値になります。
- `CMAKE_INSTALL_PREFIX: "${sourceDir}"`: コンパイル済みのコンポーネントを vLLM のソースディレクトリに戻してインストールすることを指定します。これは editable インストールにとって重要で、新しくビルドされたカーネルがすぐに Python 環境から利用できるようになります。
- ビルドプリセットの `CMAKE_JOB_POOLS` と `jobs`: ビルドの並列度を制御します。スクリプトはシステムで検出した CPU コア数にもとづいて設定します。
- `binaryDir`: ビルド成果物の格納先を指定します（`cmake-build-release` など）。

## CMake でのビルドとインストール { #building-and-installing-with-cmake }

`CMakeUserPresets.json` を設定したら、次のようにします。

1. **CMake のビルド環境を初期化する:**
   このステップでは、選んだプリセット（`release` など）に従ってビルドシステムを構成し、`binaryDir` にビルドディレクトリを作成します。

    ```console
    cmake --preset release
    ```

2. **vLLM のコンポーネントをビルドしてインストールする:**
   このコマンドはコードをコンパイルし、生成されたバイナリを vLLM のソースディレクトリにインストールして、editable な Python インストールから利用できるようにします。

    ```console
    cmake --build --preset release --target install
    ```

3. **変更して繰り返す**
    これで editable インストールした vLLM を使い、必要に応じてテストや変更を行えます。変更を反映するために再ビルドが必要になったら、CMake のコマンドをもう一度実行するだけで、影響を受けたファイルだけがビルドされます。

    ```console
    cmake --build --preset release --target install
    ```

## ビルドの確認 { #verifying-the-build }

ビルドに成功すると、ファイルが生成されたビルドディレクトリができます（`release` プリセットと上記の設定例を使った場合は `cmake-build-release/` など）。

```console
> ls cmake-build-release/
bin             cmake_install.cmake      _deps                                machete_generation.log
build.ninja     CPackConfig.cmake        detect_cuda_compute_capabilities.cu  marlin_generation.log
_C.abi3.so      CPackSourceConfig.cmake  detect_cuda_version.cc               _moe_C.abi3.so
CMakeCache.txt  ctest                    _flashmla_C.abi3.so                  moe_marlin_generation.log
CMakeFiles      cumem_allocator.abi3.so  install_local_manifest.txt           vllm-flash-attn
```

`cmake --build ... --target install` コマンドは、コンパイル済みの共有ライブラリ（`_C.abi3.so`、`_moe_C.abi3.so` など）を、ソースツリー内の適切な `vllm` パッケージディレクトリにコピーします。これにより、editable インストールが新しくコンパイルされたカーネルで更新されます。

## その他のヒント { #additional-tips }

- **並列度の調整:** `CMakeUserPresets.json` の `configurePresets` にある `CMAKE_JOB_POOLS` と `buildPresets` にある `jobs` を調整してください。ジョブ数が多すぎると、RAM や CPU コアが限られたシステムでは負荷が高くなり、ビルドが遅くなったりシステムが不安定になったりします。少なすぎるとリソースを活かしきれません。
- **必要に応じてクリーンビルドする:** 大きな変更のあとやブランチを切り替えたあとなどに、ビルドエラーが解消しない、あるいは不可解な場合は、CMake のビルドディレクトリを削除して（`rm -rf cmake-build-release` など）、`cmake --preset` と `cmake --build` を実行し直すことを検討してください。
- **特定ターゲットのビルド:** 特定のモジュールを開発しているときは、`install` ターゲット全体ではなく特定のターゲットだけをビルドすることで、さらに速く反復できる場合があります。ただし、Python 環境で必要なコンポーネントがすべて更新されることを保証するのは `install` です。より高度なターゲット管理については CMake のドキュメントを参照してください。
