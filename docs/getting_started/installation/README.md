# インストール { #installation }

vLLM は次のハードウェアプラットフォームをサポートしています。

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [Apple Silicon](gpu.md)（[vLLM-Metal](https://github.com/vllm-project/vllm-metal) 経由）
- [CPU](cpu.md)
    - [Intel/AMD x86](cpu.md#intelamd-x86)
    - [ARM AArch64](cpu.md#arm-aarch64)
    - [Apple silicon](cpu.md#apple-silicon)
    - [IBM Z (S390X)](cpu.md#ibm-z-s390x)

## ハードウェアプラグイン { #hardware-plugins }

vLLM は、`vllm` 本体のリポジトリの**外**で保守されるサードパーティ製ハードウェアプラグインをサポートしています。これらは [Hardware-Pluggable RFC](../../design/plugin_system.md) に従っています。

サポートされているハードウェアの一覧は vLLM の Web サイト [Universal Compatibility - Hardware](https://vllm.ai/#compatibility) を参照してください。

新しいハードウェアを追加したい場合は、[Slack](https://slack.vllm.ai/) または[メール](mailto:collaboration@vllm.ai)でご連絡ください。
