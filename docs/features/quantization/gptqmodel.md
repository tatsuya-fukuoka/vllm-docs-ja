# GPTQModel { #gptqmodel }

4 ビットまたは 8 ビットの GPTQ 量子化モデルを新たに作るには、ModelCloud.AI の [GPTQModel](https://github.com/ModelCloud/GPTQModel) を利用できます。

量子化はモデルの精度を BF16/FP16（16 ビット）から INT4（4 ビット）または INT8（8 ビット）へ落とし、
モデル全体のメモリ使用量を大きく削減すると同時に推論性能を高めます。

対応する GPTQModel の量子化モデルは、vLLM の独自カーネル `Marlin` と `Machete` を活用でき、
Ampere（A100 以降）と Hopper（H100 以降）の NVIDIA GPU でバッチ処理の毎秒トランザクション数 `tps` と
トークンのレイテンシ性能を最大化できます。この 2 つのカーネルは vLLM と NeuralMagic（現在は Red Hat の一部）に
よって高度に最適化されており、量子化 GPTQ モデルで世界水準の推論性能を実現します。

GPTQModel は、LLM 内の層やモジュールごとに個別の量子化パラメータでさらに最適化できる `Dynamic`（モジュール単位）量子化に対応した、
数少ない量子化ツールキットの 1 つです。`Dynamic` 量子化は vLLM に完全に統合されており、ModelCloud.AI チームのサポートを受けています。
この機能やその他の高度な機能の詳細は [GPTQModel の readme](https://github.com/ModelCloud/GPTQModel?tab=readme-ov-file#dynamic-quantization-per-module-quantizeconfig-override)（英語）を参照してください。

## インストール { #installation }

[GPTQModel](https://github.com/ModelCloud/GPTQModel) をインストールして自分のモデルを量子化することも、[Hugging Face にある 5000 以上のモデル](https://huggingface.co/models?search=gptq)から選ぶこともできます。

```bash
pip install -U gptqmodel --no-build-isolation -v
```

## モデルを量子化する { #quantizing-a-model }

GPTQModel をインストールすると、モデルを量子化できます。詳細は [GPTQModel の readme](https://github.com/ModelCloud/GPTQModel/?tab=readme-ov-file#quantization)（英語）を参照してください。

以下は `meta-llama/Llama-3.2-1B-Instruct` を量子化する例です。

??? code

    ```python
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(1024))["text"]

    quant_config = QuantizeConfig(bits=4, group_size=128)

    model = GPTQModel.load(model_id, quant_config)

    # increase `batch_size` to match gpu/vram specs to speed up quantization
    model.quantize(calibration_dataset, batch_size=2)

    model.save(quant_path)
    ```

## 量子化したモデルを vLLM で実行する { #running-a-quantized-model-with-vllm }

GPTQModel で量子化したモデルを vLLM で実行するには、[DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2](https://huggingface.co/ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2) を次のコマンドで使えます。

```bash
python examples/deployment/llm_engine_example.py \
    --model ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2
```

## vLLM の Python API から GPTQModel を使う { #using-gptqmodel-with-vllms-python-api }

GPTQModel で量子化したモデルは、LLM のエントリポイントから直接利用することもできます。

??? code

    ```python
    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

    # Create an LLM.
    llm = LLM(model="ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2")

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("-"*50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-"*50)
    ```
