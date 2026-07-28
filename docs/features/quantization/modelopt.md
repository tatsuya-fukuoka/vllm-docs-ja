# NVIDIA Model Optimizer { #nvidia-model-optimizer }

[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) は、NVIDIA GPU での推論に向けてモデルを最適化するライブラリです。大規模言語モデル (LLM)、視覚言語モデル (VLM)、拡散モデルの学習後量子化 (PTQ) と量子化を考慮した学習 (QAT) のツールを備えています。

ライブラリは次の方法でインストールすることを推奨します。

```bash
pip install nvidia-modelopt
```

## サポートする ModelOpt のチェックポイント形式 { #supported-modelopt-checkpoint-formats }

vLLM は `hf_quant_config.json` によって ModelOpt のチェックポイントを検出し、
次の `quantization.quant_algo` の値をサポートします。

- `FP8`: テンソル単位の重みスケール（＋任意で静的なアクティベーションスケール）。
- `FP8_PER_CHANNEL_PER_TOKEN`: チャネル単位の重みスケールと、トークン単位の動的なアクティベーション量子化。
- `FP8_PB_WO`（ModelOpt は `fp8_pb_wo` と出力することがあります）: ブロック単位でスケールする重みのみの FP8（通常は 128×128 のブロック）。
- `NVFP4`: ModelOpt の NVFP4 チェックポイント（`quantization="modelopt_fp4"` を指定）。
- `MXFP8`: ModelOpt の MXFP8 チェックポイント（`quantization="modelopt_mxfp8"` を指定）。

## PTQ で HuggingFace のモデルを量子化する { #quantizing-huggingface-models-with-ptq }

Model Optimizer のリポジトリにあるサンプルスクリプトを使って Hugging Face のモデルを量子化できます。LLM の PTQ 用の主なスクリプトは通常 `examples/llm_ptq` ディレクトリにあります。

以下は、modelopt の PTQ API でモデルを量子化する例です。

??? code

    ```python
    import modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM

    # Load the model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained("<path_or_model_id>")

    # Select the quantization config, for example, FP8
    config = mtq.FP8_DEFAULT_CFG

    # Define a forward loop function for calibration
    def forward_loop(model):
        for data in calib_set:
            model(data)

    # PTQ with in-place replacement of quantized modules
    model = mtq.quantize(model, config, forward_loop)
    ```

量子化が終わったら、エクスポート API で量子化済みのチェックポイントとして出力できます。

```python
import torch
from modelopt.torch.export import export_hf_checkpoint

with torch.inference_mode():
    export_hf_checkpoint(
        model,  # The quantized model.
        export_dir,  # The directory where the exported files will be stored.
    )
```

量子化済みのチェックポイントは vLLM でデプロイできます。次のコードは、`meta-llama/Llama-3.1-8B-Instruct` から派生した FP8 量子化のチェックポイントである `nvidia/Llama-3.1-8B-Instruct-FP8` を vLLM でデプロイする例です。

??? code

    ```python
    from vllm import LLM, SamplingParams

    def main():
        model_id = "nvidia/Llama-3.1-8B-Instruct-FP8"

        # Ensure you specify quantization="modelopt" when loading the modelopt checkpoint
        llm = LLM(model=model_id, quantization="modelopt", trust_remote_code=True)

        sampling_params = SamplingParams(temperature=0.8, top_p=0.9)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if __name__ == "__main__":
        main()
    ```

## OpenAI 互換サーバーを実行する { #running-the-openai-compatible-server }

ローカルの ModelOpt のチェックポイントを OpenAI 互換 API でサービングするには次のようにします。

```bash
vllm serve <path_to_exported_checkpoint> \
  --quantization modelopt \
  --host 0.0.0.0 --port 8000
```

## テスト（ローカルのチェックポイント） { #testing-local-checkpoints }

vLLM の ModelOpt のユニットテストはローカルのチェックポイントのパスに依存するため、
CI では既定でスキップされます。ローカルで実行するには次のようにします。

```bash
export VLLM_TEST_MODELOPT_FP8_PC_PT_MODEL_PATH=<path_to_fp8_pc_pt_checkpoint>
export VLLM_TEST_MODELOPT_FP8_PB_WO_MODEL_PATH=<path_to_fp8_pb_wo_checkpoint>
pytest -q tests/quantization/test_modelopt.py
```
