<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM は、Intel GPU プラットフォーム上での基本的なモデル推論とサービングを初期段階でサポートしています。

--8<-- [end:installation]
--8<-- [start:requirements]

- サポートされるハードウェア: Intel Data Center GPU、Intel ARC GPU
- 依存パッケージ: [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels) — Intel GPU プラットフォーム上で vLLM を実行する際に必要な vLLM のカスタムカーネルをすべて提供するパッケージ
- Python: 3.12
!!! warning
    提供されている vllm-xpu-kernels の whl は Python 3.12 専用のため、このバージョンが必須です。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

このデバイス向けに新しい Python 環境を作成する際の追加情報はありません。

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

現時点では、XPU 向けのビルド済み wheel はありません。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

- まず、必要な[ドライバ](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers)をインストールします。
- 次に、vLLM の XPU バックエンドをビルドするための Python パッケージをインストールします（Intel OneAPI の依存パッケージは `torch-xpu` の一部として自動的にインストールされます。[PyTorch XPU の入門](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)を参照）。
- vllm-xpu-kernels v0.1.10 以降では、互換性の問題を避けるため、ドライバを [compute runtime 26.18](https://github.com/intel/compute-runtime/releases/tag/26.14.37833.4) リリースにアップグレードすることを推奨します。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- 次に、Intel XPU 用の正しい Triton パッケージをインストールします。

    既定の `triton` パッケージ（NVIDIA GPU 向け）が推移的な依存関係として（`xgrammar` 経由などで）インストールされることがあります。Intel XPU では、これを `triton-xpu` に置き換える必要があります。

    ```bash
    pip uninstall -y triton triton-xpu
    pip install triton-xpu==3.7.1 --extra-index-url https://download.pytorch.org/whl/xpu
    ```

    !!! note
        - 接尾辞のない `triton` は NVIDIA GPU 専用です。XPU で `triton-xpu` の代わりにこれを使うと、正しさや実行時の問題を引き起こす可能性があります。
        - torch 2.12（`requirements/xpu.txt` で使われているバージョン）に対応するパッケージは `triton-xpu==3.7.1` です。別のバージョンの torch を使う場合は、[docker/Dockerfile.xpu](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.xpu) で対応する `triton-xpu` のバージョンを確認してください。

- 最後に、vLLM の XPU バックエンドをビルド・インストールします。

```bash
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

現在、vLLM のリリースバージョンにもとづくビルド済みの XPU イメージを Docker [Hub](https://hub.docker.com/r/intel/vllm/tags) で公開しています。詳細はリリース[ノート](https://github.com/intel/ai-containers/blob/main/vllm)を参照してください。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri:/dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             --ipc=host \
             --privileged \
             vllm-xpu-env
```

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

XPU プラットフォームは**テンソル並列**の推論 / サービングをサポートし、オンラインサービングではベータ機能として**パイプライン並列**もサポートしています。**パイプライン並列**については、バックエンドに mp を使う単一ノード構成をサポートしています。実行例は次のとおりです。

```bash
vllm serve facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=mp \
     --pipeline-parallel-size=2 \
     -tp=8
```

既定では、システム上に既存の ray インスタンスが検出されない場合、`num-gpus` を `parallel_config.world_size` として ray インスタンスが自動的に起動されます。実行前に [examples/ray_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/ray_serving/run_cluster.sh) のヘルパースクリプトを参考に、ray クラスタを適切に起動しておくことを推奨します。

--8<-- [end:supported-features]
--8<-- [start:distributed-backend]

XPU プラットフォームでは、分散バックエンドとして torch 2.8 未満では **torch-ccl** を、torch 2.8 以降では **xccl** を使います。torch 2.8 以降では XPU 向けの組み込みバックエンドとして **xccl** がサポートされているためです。

--8<-- [end:distributed-backend]
