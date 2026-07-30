# XPU - Intel® GPU { #xpu-intel-gpus }

## 検証済みハードウェア { #validated-hardware }

| ハードウェア |
| -------- |
| [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html) |

## 推奨モデル { #recommended-models }

### テキストのみの言語モデル { #text-only-language-models }

| モデル                                              | アーキテクチャ                                     | BF16/FP16/動的 FP8 | Compressed_tensors FP8 | MXFP4 |
| -------------------------------------------------- | ------------------------------------------------ | --------------------- | ---------------------- | ----- |
| openai/gpt-oss-20b                                 | GPTForCausalLM                                   |                       |                        | ✅    |
| openai/gpt-oss-120b                                | GPTForCausalLM                                   |                       |                        | ✅    |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B           | LlamaForCausalLM                                 | ✅                    |                        |       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B           | QwenForCausalLM                                  | ✅                    |                        |       |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B           | QwenForCausalLM                                  | ✅                    |                        |       |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B          | LlamaForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen2.5-72B-Instruct                          | Qwen2ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-14B                                     | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-32B                                     | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-30B-A3B                                 | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-30B-A3B-GPTQ-Int4                       | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/Qwen3-coder-30B-A3B-Instruct                  | Qwen3ForCausalLM                                 | ✅                    |                        |       |
| Qwen/QwQ-32B                                       | QwenForCausalLM                                  | ✅                    |                        |       |
| deepseek-ai/DeepSeek-V2-Lite                       | DeepSeekForCausalLM                              | ✅                    |                        |       |
| meta-llama/Llama-3.1-8B-Instruct                   | LlamaForCausalLM                                 | ✅                    |                        |       |
| THUDM/GLM-4-9B-chat                                | GLMForCausalLM                                   | ✅                    |                        |       |
| THUDM/CodeGeex4-All-9B                             | CodeGeexForCausalLM                              | ✅                    |                        |       |
| chuhac/TeleChat2-35B                               | LlamaForCausalLM (TeleChat2 based on Llama arch) | ✅                    |                        |       |
| 01-ai/Yi1.5-34B-Chat                               | YiForCausalLM                                    | ✅                    |                        |       |
| THUDM/CodeGeex4-All-9B                             | CodeGeexForCausalLM                              | ✅                    |                        |       |
| deepseek-ai/DeepSeek-Coder-33B-base                | DeepSeekCoderForCausalLM                         | ✅                    |                        |       |
| meta-llama/Llama-2-13b-chat-hf                     | LlamaForCausalLM                                 | ✅                    |                        |       |
| THUDM/CodeGeex4-All-9B                             | CodeGeexForCausalLM                              | ✅                    |                        |       |
| Qwen/Qwen1.5-14B-Chat                              | QwenForCausalLM                                  | ✅                    |                        |       |
| Qwen/Qwen1.5-32B-Chat                              | QwenForCausalLM                                  | ✅                    |                        |       |
| RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic    | LlamaForCausalLM                                 |                       | ✅                     |       |

### マルチモーダル言語モデル { #multimodal-language-models }

| モデル                        | アーキテクチャ                     | BF16 | 動的 FP8 | MXFP4 |
| ---------------------------- | -------------------------------- | ---- | ----------- | ----- |
| OpenGVLab/InternVL3_5-8B     | InternVLForConditionalGeneration | ✅   | ✅          |       |
| OpenGVLab/InternVL3_5-14B    | InternVLForConditionalGeneration | ✅   | ✅          |       |
| OpenGVLab/InternVL3_5-38B    | InternVLForConditionalGeneration | ✅   | ✅          |       |
| Qwen/Qwen2-VL-7B-Instruct    | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| Qwen/Qwen2.5-VL-72B-Instruct | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| Qwen/Qwen2.5-VL-32B-Instruct | Qwen2VLForConditionalGeneration  | ✅   | ✅          |       |
| THUDM/GLM-4v-9B              | GLM4vForConditionalGeneration    | ✅   | ✅          |       |
| openbmb/MiniCPM-V-4          | MiniCPMVForConditionalGeneration | ✅   | ✅          |       |

### 埋め込み・リランカー言語モデル { #embedding-and-reranker-language-models }

| モデル                   | アーキテクチャ                   | BF16 | 動的 FP8 | MXFP4 |
| ----------------------- | ------------------------------ | ---- | ----------- | ----- |
| Qwen/Qwen3-Embedding-8B | Qwen3ForTextEmbedding          | ✅   | ✅          |       |
| Qwen/Qwen3-Reranker-8B  | Qwen3ForSequenceClassification | ✅   | ✅          |       |

✅ 動作し、最適化済み。  
🟨 動作し結果も正しいが、まだ十分に最適化されていない。  
❌ 精度テストに合格しない、または動作しない。  
