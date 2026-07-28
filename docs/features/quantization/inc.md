# Intel の量子化サポート { #intel-quantization-support }

[AutoRound](https://github.com/intel/auto-round) は、大規模言語モデル (LLM) 向けに設計された Intel の高度な量子化アルゴリズムです。**INT2・INT3・INT4・INT8・MXFP8・MXFP4・NVFP4**、および **GGUF** の効率的な量子化モデルを生成し、精度と推論性能のバランスを取ります。AutoRound は [Intel® Neural Compressor](https://github.com/intel/neural-compressor) の一部でもあります。詳しい紹介は [AutoRound の手順ガイド](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md)（英語）を参照してください。

## 主な機能 { #key-features }

✅ 高い精度。2〜3 ビットでも良好な性能を発揮します（[モデルの例](https://huggingface.co/collections/OPEA/2-3-bits)）

✅ `Bits` / `Dtypes` の混合スキームを高速に生成。数分で自動設定できます

✅ **AutoRound・AutoAWQ・AutoGPTQ・GGUF** 形式でのエクスポートに対応

✅ **10 種類以上の視覚言語モデル (VLM)** に対応

✅ 細かな制御のための**層ごとの混合ビット量子化**

✅ わずかな精度低下と引き換えに素早く量子化できる **RTN（最近接丸め）モード**

✅ **複数の量子化レシピ**: best・base・light

✅ 即時パッキングなどの高度なユーティリティと、**10 種類以上のバックエンド**への対応

## Intel プラットフォームで対応するレシピ { #supported-recipes-on-intel-platforms }

Intel のプラットフォームでは、AutoRound のレシピは形式とハードウェアごとに段階的に対応が進んでいます。現時点で vLLM が対応しているのは次のとおりです。

- **`W4A16`**: 重みのみの量子化。4 ビットの重みと 16 ビットのアクティベーション
- **`W8A16`**: 重みのみの量子化。8 ビットの重みと 16 ビットのアクティベーション

その他のレシピと形式は今後のリリースで対応予定です。

## モデルを量子化する { #quantizing-a-model }

### インストール { #installation }

```bash
uv pip install auto-round
```

### CLI で量子化する { #quantize-with-cli }

```bash
auto-round \
    --model Qwen/Qwen3-0.6B \
    --scheme W4A16 \
    --format auto_round \
    --output_dir ./tmp_autoround
```

### Python API で量子化する { #quantize-with-python-api }

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "Qwen/Qwen3-0.6B"
autoround = AutoRound(model_name, scheme="W4A16")

# the best accuracy, 4-5X slower, low_gpu_mem_usage could save ~20G but ~30% slower
# autoround = AutoRound(model, tokenizer, nsamples=512, iters=1000, low_gpu_mem_usage=True, bits=bits, group_size=group_size, sym=sym)

# 2-3X speedup, slight accuracy drop at W4G128
# autoround = AutoRound(model, tokenizer, nsamples=128, iters=50, lr=5e-3, bits=bits, group_size=group_size, sym=sym )

output_dir = "./tmp_autoround"
# format= 'auto_round'(default), 'auto_gptq', 'auto_awq'
autoround.quantize_and_save(output_dir, format="auto_round")
```

## AutoRound で量子化したモデルを vLLM でデプロイする { #deploying-autoround-quantized-models-in-vllm }

```bash
vllm serve Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

## 量子化したモデルを vLLM で評価する { #evaluating-the-quantized-model-with-vllm }

```bash
lm_eval --model vllm \
  --model_args pretrained="Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound,max_model_len=8192,max_num_batched_tokens=32768,max_num_seqs=128,gpu_memory_utilization=0.8,dtype=bfloat16,max_gen_toks=2048" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size 128
```
