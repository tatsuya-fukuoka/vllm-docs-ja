# AMD Quark { #amd-quark }

量子化は、精度の低下を最小限に抑えつつ、メモリと帯域幅の使用量を効果的に削減し、計算を高速化してスループットを向上させます。vLLM は、柔軟で強力な量子化ツールキットである [Quark](https://quark.docs.amd.com/latest/) を活用して、AMD GPU 上で高い性能を発揮する量子化モデルを生成できます。Quark は、重み・活性値・KV キャッシュの量子化や、AWQ、GPTQ、Rotation、SmoothQuant といった最先端の量子化アルゴリズムによる大規模言語モデルの量子化を専門的にサポートしています。

## Quark のインストール { #quark-installation }

モデルを量子化する前に Quark をインストールする必要があります。最新版の Quark は pip でインストールできます。

```bash
pip install amd-quark
```

インストールの詳細は [Quark のインストールガイド](https://quark.docs.amd.com/latest/install.html)を参照してください。

さらに、評価のために `vllm` と `lm-evaluation-harness` をインストールします。

```bash
pip install vllm "lm-eval[api]>=0.4.12"
```

## 量子化の手順 { #quantization-process }

Quark をインストールしたら、例を使って使い方を説明します。Quark の量子化プロセスは、次の 5 ステップに整理できます。

1. モデルの読み込み
2. キャリブレーション用データローダーの準備
3. 量子化設定の指定
4. モデルの量子化とエクスポート
5. vLLM での評価

### 1. モデルの読み込み { #1-load-the-model }

Quark はモデルとトークナイザーの取得に [Transformers](https://huggingface.co/docs/transformers/en/index) を使います。

??? code

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_ID = "meta-llama/Llama-2-70b-chat-hf"
    MAX_SEQ_LEN = 512

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
    tokenizer.pad_token = tokenizer.eos_token
    ```

### 2. キャリブレーション用データローダーの準備 { #2-prepare-the-calibration-dataloader }

Quark はキャリブレーションデータの読み込みに [PyTorch の Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) を使います。キャリブレーション用データセットを効率的に使う方法の詳細は、[キャリブレーションデータセットの追加](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html)を参照してください。

??? code

    ```python
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    BATCH_SIZE = 1
    NUM_CALIBRATION_DATA = 512

    # Load the dataset and get calibration data.
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    text_data = dataset["text"][:NUM_CALIBRATION_DATA]

    tokenized_outputs = tokenizer(
        text_data,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    calib_dataloader = DataLoader(
        tokenized_outputs['input_ids'],
        batch_size=BATCH_SIZE,
        drop_last=True,
    )
    ```

### 3. 量子化設定の指定 { #3-set-the-quantization-configuration }

量子化の設定を指定する必要があります。詳細は [Quark の設定ガイド](https://quark.docs.amd.com/latest/pytorch/user_guide_config_description.html)を確認してください。ここでは、重み・活性値・KV キャッシュにテンソル単位の FP8 量子化を使い、量子化アルゴリズムには AutoSmoothQuant を使います。

!!! note
    量子化アルゴリズムには JSON の設定ファイルが必要で、その設定ファイルは
    [Quark の PyTorch サンプル](https://quark.docs.amd.com/latest/pytorch/pytorch_examples.html)の
    `examples/torch/language_modeling/llm_ptq/models` ディレクトリにあります。たとえば Llama 向けの
    AutoSmoothQuant の設定ファイルは
    `examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json` です。

??? code

    ```python
    from quark.torch.quantization import (Config, QuantizationConfig,
                                        FP8E4M3PerTensorSpec,
                                        load_quant_algo_config_from_file)

    # Define fp8/per-tensor/static spec.
    FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(
        observer_method="min_max",
        is_dynamic=False,
    ).to_quantization_spec()

    # Define global quantization config, input tensors and weight apply FP8_PER_TENSOR_SPEC.
    global_quant_config = QuantizationConfig(
        input_tensors=FP8_PER_TENSOR_SPEC,
        weight=FP8_PER_TENSOR_SPEC,
    )

    # Define quantization config for kv-cache layers, output tensors apply FP8_PER_TENSOR_SPEC.
    KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
    kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
    kv_cache_quant_config = {
        name: QuantizationConfig(
            input_tensors=global_quant_config.input_tensors,
            weight=global_quant_config.weight,
            output_tensors=KV_CACHE_SPEC,
        )
        for name in kv_cache_layer_names_for_llama
    }
    layer_quant_config = kv_cache_quant_config.copy()

    # Define algorithm config by config file.
    LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE = "examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json"
    algo_config = load_quant_algo_config_from_file(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)

    EXCLUDE_LAYERS = ["lm_head"]
    quant_config = Config(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        exclude=EXCLUDE_LAYERS,
        algo_config=algo_config,
    )
    ```

### 4. モデルの量子化とエクスポート { #4-quantize-the-model-and-export }

続いて量子化を適用します。量子化のあと、エクスポートの前にまず量子化済みモデルを freeze する必要があります。モデルは HuggingFace の `safetensors` 形式でエクスポートする必要がある点に注意してください。エクスポート形式の詳細は [HuggingFace 形式でのエクスポート](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html)を参照してください。

??? code

    ```python
    import torch
    from quark.torch import ModelQuantizer, ModelExporter
    from quark.torch.export import ExporterConfig, JsonExporterConfig

    # Apply quantization.
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, calib_dataloader)

    # Freeze quantized model to export.
    freezed_model = quantizer.freeze(model)

    # Define export config.
    LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
    export_config = ExporterConfig(json_export_config=JsonExporterConfig())
    export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP

    # Model: Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant
    EXPORT_DIR = MODEL_ID.split("/")[1] + "-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant"
    exporter = ModelExporter(config=export_config, export_dir=EXPORT_DIR)
    with torch.no_grad():
        exporter.export_safetensors_model(
            freezed_model,
            quant_config=quant_config,
            tokenizer=tokenizer,
        )
    ```

### 5. vLLM での評価 { #5-evaluation-in-vllm }

これで、Quark で量子化したモデルを LLM エントリポイントから直接読み込んで実行できます。

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
    llm = LLM(
        model="Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant",
        kv_cache_dtype="fp8",
        quantization="quark",
    )
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)
    ```

あるいは、`lm_eval` を使って精度を評価できます。

```bash
lm_eval --model vllm \
  --model_args pretrained=Llama-2-70b-chat-hf-w-fp8-a-fp8-kvcache-fp8-pertensor-autosmoothquant,kv_cache_dtype='fp8',quantization='quark' \
  --tasks gsm8k
```

## Quark の量子化スクリプト { #quark-quantization-script }

上記の Python API の例に加えて、Quark は大規模言語モデルをより手軽に量子化するための[量子化スクリプト](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)も提供しています。このスクリプトはさまざまな量子化方式と最適化アルゴリズムでのモデル量子化に対応し、量子化済みモデルのエクスポートと評価タスクの実行をその場で行えます。このスクリプトを使うと、上記の例は次のように書けます。

```bash
python3 quantize_quark.py --model_dir meta-llama/Llama-2-70b-chat-hf \
                          --output_dir /path/to/output \
                          --quant_scheme w_fp8_a_fp8 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --num_calib_data 512 \
                          --model_export hf_format \
                          --tasks gsm8k
```

## OCP MX（MXFP4、MXFP6）モデルの利用 { #using-ocp-mx-mxfp4-mxfp6-models }

vLLM は、[Open Compute Project（OCP）の仕様](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)に準拠し、AMD Quark でオフラインに量子化された MXFP4 および MXFP6 のモデルの読み込みをサポートしています。

この方式は現時点で、活性値については動的量子化のみをサポートしています。

最新版の AMD Quark をインストールしたあとの使用例は次のとおりです。

```bash
vllm serve fxmarty/qwen_1.5-moe-a2.7b-mxfp4 --tensor-parallel-size 1
# or, for a model using fp6 activations and fp4 weights:
vllm serve fxmarty/qwen1.5_moe_a2.7b_chat_w_fp4_a_fp6_e2m3 --tensor-parallel-size 1
```

OCP MX の演算をネイティブにサポートしないデバイス（AMD Instinct MI325、MI300、MI250 など）でも、融合カーネルを使って重みを FP4/FP6 からその場で半精度に逆量子化することで、MXFP4/MXFP6 での行列積のシミュレーション実行が可能です。これは、vLLM で FP4/FP6 モデルを評価したい場合や、（float16 や bfloat16 と比べて）約 2.5〜4 倍のメモリ削減の恩恵を受けたい場合に有用です。

MXFP4 のデータ型で量子化したオフラインのモデルを生成するには、AMD Quark の[量子化スクリプト](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)を使うのが最も簡単です。例を示します。

```bash
python quantize_quark.py --model_dir Qwen/Qwen1.5-MoE-A2.7B-Chat \
    --quant_scheme w_mxfp4_a_mxfp4 \
    --output_dir qwen_1.5-moe-a2.7b-mxfp4 \
    --skip_evaluation \
    --model_export hf_format \
    --group_size 32
```

現在の統合では、重みまたは活性値に使う [FP4、FP6_E3M2、FP6_E2M3 のすべての組み合わせ](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py)がサポートされています。

## Quark で量子化した層ごとの自動混合精度（AMP）モデルの利用 { #using-quark-quantized-layerwise-auto-mixed-precision-amp-models }

vLLM は、AMD Quark で量子化された層ごとの混合精度モデルの読み込みもサポートしています。現時点でサポートされる混合方式は {MXFP4, FP8} で、ここでの FP8 はテンソル単位の FP8 方式を指します。近い将来、次のような混合精度方式のサポートも予定されています。

- 各層の選択肢として、量子化しない Linear 層や MoE 層を許容する、つまり {MXFP4, FP8, BF16/FP16} の混合
- MXFP6 量子化への拡張、つまり {MXFP4, MXFP6, FP8, BF16/FP16}

デバイスがサポートする最も低い精度（AMD Instinct MI355 なら MXFP4、AMD Instinct MI300 なら FP8 など）を使えばサービングのスループットを最大化できますが、こうした積極的な方式は、対象タスクにおける量子化からの精度回復を妨げることがあります。混合精度を使えば、精度とスループットの最大化のバランスを取れます。

AMD Quark で量子化した混合精度モデルを生成・デプロイする手順は、次の 2 ステップです。

### 1. AMD Quark で混合精度を使ってモデルを量子化する { #1-quantize-a-model-using-mixed-precision-in-amd-quark }

まず、対象の LLM モデルについて層ごとの混合精度の構成を探索し、AMD Quark で量子化します。Quark の API を使った詳細なチュートリアルは後日提供予定です。

vLLM での使い方と精度面の利点を示すため、すぐに使える量子化済みの混合精度モデルをいくつか用意しています。

- amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
- amd/Mixtral-8x7B-Instruct-v0.1-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8
- amd/Qwen3-8B-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8

### 2. 量子化した混合精度モデルを vLLM で推論する { #2-inference-the-quantized-mixed-precision-model-in-vllm }

AMD Quark で混合精度を使って量子化したモデルは、vLLM でそのまま読み込めます。たとえば lm-evaluation-harness で次のように評価できます。

```bash
lm_eval --model vllm \
    --model_args pretrained=amd/Llama-2-70b-chat-hf-WMXFP4FP8-AMXFP4FP8-AMP-KVFP8,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.8,trust_remote_code=False \
    --tasks mmlu \
    --batch_size auto
```
