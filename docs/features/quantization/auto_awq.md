# AutoAWQ { #autoawq }

> ⚠️ **注意:**
    `AutoAWQ` ライブラリは非推奨です。この機能は vLLM プロジェクトの [`llm-compressor`](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) に取り込まれました。
    推奨される量子化のワークフローは [`llm-compressor`](https://github.com/vllm-project/llm-compressor/tree/main/examples/awq) の AWQ の例を参照してください。非推奨化の詳細は元の [AutoAWQ のリポジトリ](https://github.com/casper-hansen/AutoAWQ)を参照してください。

4 ビット量子化のモデルを新たに作るには [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) を利用できます。
量子化はモデルの精度を BF16/FP16 から INT4 に落とし、モデル全体のメモリ使用量を効果的に削減します。
主な利点はレイテンシとメモリ使用量の低減です。

AutoAWQ をインストールして自分のモデルを量子化することも、[Hugging Face にある 6500 以上のモデル](https://huggingface.co/models?search=awq)から選ぶこともできます。

```bash
pip install autoawq
```

AutoAWQ をインストールすると、モデルを量子化できます。詳細は [AutoAWQ のドキュメント](https://casper-hansen.github.io/AutoAWQ/examples/#basic-quantization)（英語）を参照してください。以下は `mistralai/Mistral-7B-Instruct-v0.2` を量子化する例です。

??? code

    ```python
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    quant_path = "mistral-instruct-v0.2-awq"
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')
    ```

AWQ のモデルを vLLM で実行するには、[TheBloke/Llama-2-7b-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-AWQ) を次のコマンドで使えます。

```bash
python examples/deployment/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization auto_awq
```

AWQ のモデルは、LLM のエントリポイントから直接利用することもできます。

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
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="auto_awq")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```
