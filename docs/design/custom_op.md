# CustomOp { #customop }

`CustomOp` は、さまざまな演算の forward メソッドを適切なバックエンドへディスパッチするための抽象クラスです。vLLM と OOT（Out-Of-Tree）プラグインの双方が、独自の演算を登録するための仕組みも提供します。

このドキュメントでは、vLLM における CustomOp の仕組みと、新しい `CustomOp` の実装方法を説明します。

## vLLM における CustomOp の仕組み { #how-customop-works-in-vllm }

`CustomOp` はクラス内で、すべてのカスタム op（登録名で索引される op クラス）の辞書を、vLLM 用と OOT プラグイン用の 2 つ管理しています。

`@CustomOp.register("op_name")` を使って、op クラスを `CustomOp` の仕組みに登録できます。これにより、`op_name` とそのクラスが `op_registry` の辞書に追加されます。さらに、`@CustomOp.register_oot("op_name")` によって OOT の op を登録することもできます。この仕組みについては後ほど詳しく説明します。

`CustomOp` が呼び出される（つまり `forward()` メソッドが呼ばれる）とき、それが有効であれば（つまり `--compilation_config.custom_ops '["+op_name"]'` が指定されていれば）、`current_platform` に応じて forward メソッドを適切なバックエンドへ自動的にディスパッチします。無効な場合は、`forward_native()` メソッドのみを呼び出し、この forward メソッドの PyTorch ネイティブ実装を使います。

- **CPU プラットフォーム:** `forward_cpu()` にディスパッチします。
- **CUDA プラットフォーム:** `forward_cuda()` にディスパッチします。
- **ROCm プラットフォーム:** `forward_hip()` にディスパッチします。`forward_hip()` が実装されていない場合は、`forward_cuda()` にフォールバックします。
- **XPU プラットフォーム:** `forward_xpu()` にディスパッチします。
- **TPU プラットフォーム:** `forward_tpu()` にディスパッチします。
- **OOT プラットフォーム:** `forward_oot()` にディスパッチします。これは OOT プラットフォームでのみ呼ばれます。
- **既定:** すべてのプラットフォームにおける最終的なフォールバックとして `forward_native()` にディスパッチします。

!!! note
    クラスの継承により、このディスパッチのロジックが絶対とは限らない点に注意してください。派生クラスが挙動を上書きすることがあります。

さらに vLLM は、`compilation_config.custom_ops` にもとづいて `CustomOp` を有効にするか無効にするかを決定します。具体的には、`CustomOp` が `compilation_config.custom_ops` に登録されていない場合（つまり既定の設定を使う場合）、`compilation_config.custom_ops` に `all` が含まれていれば有効になり、`none` が含まれていれば無効になります。

!!! note
    `compilation_config.custom_ops` に `all` と `none` を同時に指定することはできません。

既定では、`compilation_config.backend == "inductor"` かつ `compilation_config.mode != CompilationMode.NONE` の場合は `compilation_config.custom_ops` に `none` が追加され、そうでない場合は `all` が追加されます。言い換えると、torch compile モードで実行する場合、一部のプラットフォーム（`torch.compile` の既定バックエンドとして `inductor` を使うもの）では `CustomOp` が無効になります。この場合、無効化されたカスタム op について Inductor が（融合された）Triton カーネルを生成します。

!!! note
    マルチモーダルモデルでは、ViT 部分の性能を高めるためにデバイス固有の高度に最適化されたカーネルを使えるよう、
    vLLM は `MMEncoderAttention` や `ApplyRotaryEmb` など一部のカスタム op の有効化を強制しています。
    `CustomOp` の `__init__()` メソッドに `enforce_enable=True` を渡すことで、オブジェクト単位で強制的に有効化することもできます。

    なお、この `enforce_enable` の仕組みは、マルチモーダル部分に独立した `compilation_config` を追加したあとに削除される予定です。

## CustomOp の設定をカスタマイズする方法 { #how-to-customise-your-configuration-for-customop }

vLLM は、サーバー起動時に `--compilation_config.custom_ops '["..."]'` を手動で渡すことで、どのカスタム op を有効 / 無効にするかを細かく制御する手段も提供しています。

例:

- `--compilation_config.custom_ops '["all"]'` — すべてのカスタム op を有効にします。
- `--compilation_config.custom_ops '["none"]'` — すべてのカスタム op を無効にします。
- `--compilation_config.custom_ops '["all,-op1"]'` — op1 を除くすべてのカスタム op を有効にします（`-` を前置すると「無効」の意味）。
- `--compilation_config.custom_ops '["none,+op1,+op2"]'` — op1 と op2 のみを有効にします（`+` を前置すると「有効」の意味）。

## vLLM がサポートする CustomOp の種類 { #types-of-supported-customop-in-vllm }

**1. Attention:**

```python
--8<-- "vllm/model_executor/layers/mla.py:multi_head_latent_attention"

```

**2. 活性化関数:**

```python
--8<-- "vllm/model_executor/layers/activation.py:silu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:mul_and_silu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_new"

--8<-- "vllm/model_executor/layers/activation.py:gelu_fast"

--8<-- "vllm/model_executor/layers/activation.py:quick_gelu"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:gelu_and_mul_sparse"

--8<-- "vllm/model_executor/layers/activation.py:relu2"

--8<-- "vllm/model_executor/layers/activation.py:xielu"

--8<-- "vllm/model_executor/layers/activation.py:swigluoai_and_mul"

--8<-- "vllm/model_executor/layers/activation.py:fatrelu_and_mul"
```

**3. MM-Conv:**

```python
--8<-- "vllm/model_executor/layers/conv.py:conv2d"

--8<-- "vllm/model_executor/layers/conv.py:conv3d"
```

**4. 埋め込み:**

```python
--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:vocab_parallel_embedding"

--8<-- "vllm/model_executor/layers/vocab_parallel_embedding.py:parallel_lm_head"
```

**5. Linear:**

```python
--8<-- "vllm/model_executor/layers/linear.py:row_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:column_parallel_linear"

--8<-- "vllm/model_executor/layers/linear.py:replicated_linear"
```

**6. Logits Processor:**

```python
--8<-- "vllm/model_executor/layers/logits_processor.py:logits_processor"
```

**7. Mamba:**

```python
--8<-- "vllm/model_executor/layers/mamba/mamba_mixer.py:mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mamba_mixer2"

--8<-- "vllm/model_executor/layers/mamba/mamba_mixer2.py:mixer2_gated_rms_norm"

--8<-- "vllm/model_executor/models/plamo2.py:plamo2_mamba_mixer"

--8<-- "vllm/model_executor/layers/mamba/short_conv.py:short_conv"
```

**8. MoE:**

```python
# このコードは上流のソースを参照してください: https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/layers/fused_moe/layer.py

--8<-- "vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py:modular_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:unquantized_fused_moe"

--8<-- "vllm/model_executor/models/transformers/moe.py:transformers_fused_moe"

--8<-- "vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:grouped_topk"
```

**9. 正規化:**

```python
--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm"

--8<-- "vllm/model_executor/layers/layernorm.py:rms_norm_gated"

--8<-- "vllm/model_executor/layers/layernorm.py:gemma_rms_norm"
```

**10. 量子化:**

```python
--8<-- "vllm/model_executor/layers/quantization/input_quant_fp8.py:quant_fp8"
```

**11. RoPE:**

```python
--8<-- "vllm/model_executor/layers/rotary_embedding/base.py:rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py:dual_chunk_rotary_embedding"

--8<-- "vllm/model_executor/layers/rotary_embedding/common.py:apply_rotary_emb"
```

**12. エンコーダ:**

```python
--8<-- "vllm/model_executor/models/deepencoder2.py:qwen2_decoder"

--8<-- "vllm/model_executor/layers/attention/mm_encoder_attention.py:mm_encoder_attn"

--8<-- "vllm/model_executor/models/deepencoder.py:rel_pos_attention"
```

## 新しい CustomOp を実装する際の指針 { #guidelines-for-implementing-a-new-customop }

### vLLM で新しい CustomOp を実装する { #implement-a-new-customop-in-vllm }

ここでは、vLLM で新しい `CustomOp` を実装する方法を説明します。

手順:

1. `CustomOp` 基底クラスを継承した新しい op クラスを実装します。
2. その op クラスに `@CustomOp.register("op_name")` デコレータを付け、`CustomOp` の仕組みに登録します。
3. 必要に応じて各種の `forward_xxx()` メソッドを実装します。

`MMEncoderAttention` を例に説明します。

??? code

    ```python
    @CustomOp.register("mm_encoder_attn")
    class MMEncoderAttention(CustomOp):

        def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
            multimodal_config: MultiModalConfig | None = None,
        ) -> None:
            super().__init__()
            # Init...

        def forward_native(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call TORCH_SDPA implementation...

        def forward_cuda(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call FA or TORCH_SDPA implementation...

        def forward_cpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call TORCH_SDPA implementation...

        def forward_xpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call FA implementation...

        def forward_tpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
        ) -> torch.Tensor:
            # Call PALLAS implementation...
    ```

### OOT デバイスプラグインで新しい CustomOp を登録する { #register-a-new-customop-in-oot-device-plugins }

現在、[vLLM のハードウェアプラグインの仕組み](./plugin_system.md)のおかげで、vLLM をさまざまなハードウェア上でシームレスに動かすための OOT デバイスプラグインが数多く登場しています。この仕組みの詳細は [Introducing vLLM Hardware Plugin, Best Practice from Ascend NPU](https://blog.vllm.ai/2025/05/12/hardware-plugin.html) でも紹介されています。

- **公式のデバイスプラグイン:** [vllm-ascend](https://github.com/vllm-project/vllm-ascend)（Huawei Ascend NPU 向け）、[vllm-spyre](https://github.com/vllm-project/vllm-spyre)（Spyre 向け）、[vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)（Intel Gaudi 向け）、[vllm-neuron](https://github.com/vllm-project/vllm-neuron)（AWS Neuron 向け）、[vllm-meta](https://github.com/vllm-project/vllm-metal)（Apple Silicon 向け）など。
- **非公式のデバイスプラグイン:** [vllm-metax](https://github.com/MetaX-MACA/vLLM-metax)（MetaX GPU 向け）、[vllm-kunlun](https://github.com/baidu/vLLM-Kunlun)（Baidu Kunlun XPU 向け）、[vllm-musa](https://github.com/MooreThreads/vllm-musa)（Moore Threads GPU 向け）など。

このとき `CustomOp` を使えば、これらのハードウェアベンダーは OOT の `CustomOp` を登録して `forward_oot()` メソッドを実装するだけで、vLLM の演算を実行時にデバイス固有の高度に最適化されたカーネルへシームレスに置き換えられます。

ここからは、デバイスプラグイン向けに OOT の `CustomOp` を登録する方法を説明します。

`MMEncoderAttention` を例に説明します。

1. `MMEncoderAttention` を継承した `CustomMMEncoderAttention` クラスを実装し、その `forward_oot()` メソッドを実装します。
2. `MMEncoderAttention` を置き換えるために、`CustomMMEncoderAttention` を vLLM に登録します。

??? code

    ```python
    from vllm.model_executor.layers.attention import MMEncoderAttention
    from vllm.model_executor.custom_op import CustomOp


    @CustomOp.register_oot("MMEncoderAttention")
    class CustomMMEncoderAttention(MMEncoderAttention):

        def __init__(...):
            super().__init__(...)

        def forward_oot(...):
            # Call optimized device-specific kernels.
            ...
    ```

この場合、`op_registry_oot` に新しい項目 `{"MMEncoderAttention": CustomMMEncoderAttention}` が追加されます。`MMEncoderAttention` の op オブジェクトを初期化する際、そのクラス名（`MMEncoderAttention`）が `op_registry_oot` のキーに含まれていれば、vLLM はそれを登録済みのクラス（`CustomMMEncoderAttention`）に置き換えてインスタンス化します。

以降、この `MMEncoderAttention` の op が呼ばれると、それが有効であれば自分の `forward_oot()` が呼ばれます。これにより、vLLM を直接変更することなく、自分のハードウェア上で期待どおりの性能を得られます。

さらに、管理しやすくするため、すべての `CustomOp` を 1 か所でまとめて登録することもできます。

??? code

    ```python
    from vllm.model_executor.custom_op import CustomOp


    REGISTERED_CUSTOM_OPS = {
        "CustomOP1": YourCustomOp1,
        "CustomOP2": YourCustomOp2,
        "CustomOP3": YourCustomOp3,
    }

    for op_name, op_cls in REGISTERED_CUSTOM_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=op_name)
    ```
