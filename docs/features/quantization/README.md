# 量子化 { #quantization }

量子化は、モデルの精度と引き換えにメモリ使用量を削減し、大きなモデルをより幅広いデバイスで実行できるようにする手法です。

!!! tip
    量子化を始めるには [LLM Compressor](llm_compressor/README.md) を参照してください。vLLM でのデプロイ向けにモデルを最適化するライブラリで、FP8・INT8・INT4 などの量子化形式に対応しています。

vLLM がサポートする量子化形式は次のとおりです。

- [AutoAWQ](auto_awq.md)
- [BitsAndBytes](bnb.md)
- [GPTQModel](gptqmodel.md)
- [Intel Neural Compressor](inc.md)
- [LLM Compressor](llm_compressor/README.md)
    - [FP8 W8A8](llm_compressor/fp8.md)
    - [INT4 W4A16](llm_compressor/int4.md)
    - [INT8 W4A8](llm_compressor/int8_w4a8.md)
    - [INT8 W8A8](llm_compressor/int8_w8a8.md)
- [NVIDIA Model Optimizer](modelopt.md)
- [オンライン量子化](online.md)
- [AMD Quark](quark.md)
- [KV キャッシュの量子化](quantized_kvcache.md)
- [TorchAO](torchao.md)
- [FP8 ViT Encoder Attention](fp8_vit_attn.md)

## サポートされているハードウェア { #supported-hardware }

以下の表は、vLLM における各量子化実装とハードウェアプラットフォームの互換性を示しています。

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| 実装                      | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
| ------------------------- | ----- | ------ | ------ | --- | ------ | ------- | --------- | ------- | ------- |
| AWQ                       | ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| GPTQ                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ✅︎        | ✅︎      | ❌      |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌    | ✅︎*    | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| llm-compressor INT8 (W8A8)| ❌    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ✅︎      | ✅︎      |
| llm-compressor INT8 (W4A8)| ❌    | ❌     | ❌     | ❌  | ❌     | ❌      | ❌        | ❌      | ✅︎      |
| llm-compressor FP8 (W8A8) | ❌    | ❌     | ❌     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |
| bitsandbytes              | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| DeepSpeedFP               | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ❌      | ❌        | ❌      | ❌      |
| GGUF                      | ✅︎    | ✅︎     | ✅︎     | ✅︎  | ✅︎     | ✅︎      | ❌        | ❌      | ❌      |

- Volta は SM 7.0、Turing は SM 7.5、Ampere は SM 8.0/8.6、Ada は SM 8.9、Hopper は SM 9.0 を指します。
- ✅︎ は、その量子化方式が該当ハードウェアでサポートされていることを示します。
- ❌ は、その量子化方式が該当ハードウェアでサポートされていないことを示します。
- Intel Gaudi の量子化サポートはすべて [vLLM-Gaudi](https://github.com/vllm-project/vllm-gaudi) に移行しました。
- \* Turing は Marlin MXFP4 をサポートしていません。

!!! note
    Google TPU での量子化のサポート状況については、[TPU-Inference Recommended Models and Features](https://docs.vllm.ai/projects/tpu/en/latest/recommended_models_features/)（英語）を参照してください。

!!! note
    この互換性の表は、vLLM が対応ハードウェアや量子化方式を拡張していくのに伴って変更されることがあります。

    ハードウェアのサポート状況と量子化方式の最新情報は、[vllm/model_executor/layers/quantization](../../../vllm/model_executor/layers/quantization) を参照するか、vLLM の開発チームに問い合わせてください。

## ツリー外の量子化プラグイン { #out-of-tree-quantization-plugins }

vLLM は `@register_quantization_config` デコレータを使って、ツリー外の独自量子化方式を登録できます。これにより、vLLM のコードベースを変更せずに独自の量子化スキームを実装・利用できます。

### 独自の量子化方式を登録する { #registering-a-custom-quantization-method }

独自の量子化方式を登録するには、`QuantizationConfig` を継承したクラスを作成し、`@register_quantization_config` を付けます。`get_quant_method` は層の種類に応じて適切な量子化メソッドへ振り分けます。

```python
import torch
from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    """Custom quantization config."""

    def get_name(self) -> str:
        return "my_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum GPU compute capability, -1 for no restriction
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        # Config files to search for in model directory
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MyQuantConfig":
        # Create config from model's quantization config
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Dispatch based on layer type
        # NOTE: you only need to implement methods you care about
        if isinstance(layer, LinearBase):
            return MyQuantLinearMethod()
        elif isinstance(layer, FusedMoE):
            return MyQuantMoEMethod(layer.moe_config)
        return None
```

### QuantizationConfig で実装が必要なメソッド { #required-quantizationconfig-methods }

独自の `QuantizationConfig` サブクラスでは、次の抽象メソッドを実装する必要があります。

| メソッド | 説明 |
| ------ | ----------- |
| `get_name()` | 量子化方式の名前を返す |
| `get_supported_act_dtypes()` | サポートするアクティベーションの dtype の一覧を返す（例: `torch.float16`） |
| `get_min_capability()` | 必要な GPU の最小 compute capability を返す（例: Ampere なら 80、制限なしなら -1） |
| `get_config_filenames()` | モデルディレクトリ内で探す設定ファイル名の一覧を返す |
| `from_config(config)` | モデルの量子化設定の辞書から設定を生成するクラスメソッド |
| `get_quant_method(layer, prefix)` | 指定した層に対する量子化メソッドを返す。スキップする場合は `None` |

### 量子化された Linear メソッドを実装する { #implementing-a-quantized-linear-method }

Linear 層については、`get_quant_method` から `QuantizeMethodBase` のサブクラスを返します。出発点として `UnquantizedLinearMethod` を継承できます。

```python
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class MyQuantLinearMethod(UnquantizedLinearMethod):
    """Custom quantization method for linear layers."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        # Create quantized weights for the layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Apply custom quantization logic here
        ...
```

### 量子化された MoE メソッドを実装する { #implementing-a-quantized-moe-method }

Mixture of Experts (MoE) モデルについては、`get_quant_method` から `FusedMoEMethodBase` のサブクラスを返します。MoE の量子化をスキップしたい場合は `UnquantizedFusedMoEMethod` を利用できます。

```python
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

class MyQuantMoEMethod(FusedMoEMethodBase):
    """Custom quantization method for MoE layers."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Create quantized weights for the MoE layer
        ...

    def apply(
        self,
        layer: torch.nn.Module,
        router: "FusedMoERouter",
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Apply MoE computation with quantized weights
        ...

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # Return the MoE quantization configuration
        ...
```

参考として、`vllm/model_executor/layers/quantization/fp8.py` の `Fp8MoEMethod` など既存の実装を参照してください。

### プラグインを使う { #using-the-plugin }

登録すると、独自の量子化方式を vLLM で使えるようになります。

```python
# Register your quantization method (import the module containing your config)
import my_quant_plugin

from vllm import LLM

# Use the custom quantization method
llm = LLM(model="your-model", quantization="my_quant")
```

プラグインシステムの詳細は[プラグインシステムのドキュメント](../../design/plugin_system.md)を参照してください。
