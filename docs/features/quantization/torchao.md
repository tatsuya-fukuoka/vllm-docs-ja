# TorchAO { #torchao }

TorchAO は PyTorch 向けのアーキテクチャ最適化ライブラリで、推論と学習のための高性能なデータ型・最適化手法・カーネルを提供します。torch.compile や FSDP といった PyTorch 標準の機能と組み合わせられる点が特徴です。ベンチマークの数値は[こちら](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks)（英語）にあります。

最新の torchao の nightly を次の方法でインストールすることを推奨します。

```bash
# Install the latest TorchAO nightly build
# Choose the CUDA version that matches your system (cu126, cu128, etc.)
pip install \
    --pre torchao>=10.0.0 \
    --index-url https://download.pytorch.org/whl/nightly/cu126
```

## HuggingFace のモデルを量子化する { #quantizing-huggingface-models }

torchao を使えば、[transformers](https://huggingface.co/docs/transformers/main/en/quantization/torchao) や [diffusers](https://huggingface.co/docs/diffusers/en/quantization/torchao) のモデルを自分で量子化し、[このように](https://huggingface.co/jerryzh168/llama3-8b-int8wo) チェックポイントを Hugging Face Hub に保存できます。以下はそのコード例です。

??? code

    ```Python
    import torch
    from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import Int8WeightOnlyConfig

    model_name = "meta-llama/Meta-Llama-3-8B"
    quantization_config = TorchAoConfig(Int8WeightOnlyConfig())
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    hub_repo = # YOUR HUB REPO ID
    tokenizer.push_to_hub(hub_repo)
    quantized_model.push_to_hub(hub_repo, safe_serialization=False)
    ```

あるいは、簡単な UI でモデルを量子化できる [TorchAO Quantization space](https://huggingface.co/spaces/medmekk/TorchAO_Quantization) も利用できます。
