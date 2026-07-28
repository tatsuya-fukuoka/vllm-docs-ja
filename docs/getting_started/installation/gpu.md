---
toc_depth: 3
---

# GPU { #gpu }

vLLM は次の GPU 種別をサポートする Python ライブラリです。ベンダー固有の手順を見るには、お使いの GPU の種類を選択してください。

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:installation"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:installation"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:installation"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:installation"

## 要件 { #requirements }

- OS: Linux
- Python: 3.10 -- 3.13

!!! note
    vLLM は Windows をネイティブにはサポートしていません。Windows で vLLM を実行するには、対応する Linux ディストリビューションを入れた Windows Subsystem for Linux (WSL) を使うか、コミュニティが保守するフォーク（例: [https://github.com/SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows)）を利用してください。

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:requirements"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:requirements"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:requirements"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:requirements"

## Python でセットアップする { #set-up-using-python }

### 新しい Python 環境を作成する { #create-a-new-python-environment }

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:set-up-using-python"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:set-up-using-python"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:set-up-using-python"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:set-up-using-python"

### ビルド済み wheel {#pre-built-wheels}

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-wheels"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-wheels"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-wheels"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:pre-built-wheels"

### ソースから wheel をビルドする { #build-wheel-from-source }

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-wheel-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-wheel-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-wheel-from-source"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:build-wheel-from-source"

## Docker でセットアップする { #set-up-using-docker }

### ビルド済みイメージ { #pre-built-images }

--8<-- [start:pre-built-images]

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:pre-built-images"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:pre-built-images"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:pre-built-images"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:pre-built-images"

--8<-- [end:pre-built-images]

### ソースからイメージをビルドする { #build-image-from-source }

--8<-- [start:build-image-from-source]

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:build-image-from-source"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:build-image-from-source"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:build-image-from-source"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:build-image-from-source"

--8<-- [end:build-image-from-source]

## サポートされている機能 { #supported-features }

=== "NVIDIA CUDA"

    --8<-- "docs/getting_started/installation/gpu.cuda.inc.md:supported-features"

=== "AMD ROCm"

    --8<-- "docs/getting_started/installation/gpu.rocm.inc.md:supported-features"

=== "Intel XPU"

    --8<-- "docs/getting_started/installation/gpu.xpu.inc.md:supported-features"

=== "Apple Silicon"

    --8<-- "docs/getting_started/installation/gpu.apple.inc.md:supported-features"
