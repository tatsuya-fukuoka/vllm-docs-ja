# 翻訳状況

追随中の上流バージョン: **v0.26.0**（最終同期: 2026-07-28）

全 226 ページ中 **171 ページ**を翻訳済みです（75.7%）。

未翻訳のページは英語原文をそのまま掲載しています。
翻訳の追加・修正は [GitHub リポジトリ](https://github.com/tatsuya-fukuoka/vllm-docs-ja) へ Pull Request を歓迎します。

## セクション別

| セクション | 翻訳済み | 要更新 | 未翻訳 | 合計 |
| --- | ---: | ---: | ---: | ---: |
| トップページ | 1 | 0 | 0 | 1 |
| ベンチマーク | 2 | 0 | 2 | 4 |
| CLI リファレンス | 15 | 0 | 0 | 15 |
| コミュニティ | 3 | 0 | 0 | 3 |
| 設定 | 7 | 0 | 0 | 7 |
| 開発者ガイド | 11 | 0 | 5 | 16 |
| デプロイ | 35 | 0 | 0 | 35 |
| 設計ドキュメント | 8 | 0 | 21 | 29 |
| 機能 | 40 | 0 | 12 | 52 |
| はじめに | 5 | 0 | 8 | 13 |
| ガバナンス | 3 | 0 | 0 | 3 |
| モデル | 10 | 0 | 6 | 16 |
| 推論とサービング | 15 | 0 | 1 | 16 |
| 学習 | 8 | 0 | 0 | 8 |
| 使い方 | 8 | 0 | 0 | 8 |
| **合計** | **171** | **0** | **55** | **226** |

## 翻訳済みのページ

- `README.md`
- `benchmarking/README.md`
- `benchmarking/dashboard.md`
- `cli/README.md`
- `cli/bench/latency.md`
- `cli/bench/mm_processor.md`
- `cli/bench/serve.md`
- `cli/bench/sweep/plot.md`
- `cli/bench/sweep/plot_pareto.md`
- `cli/bench/sweep/serve.md`
- `cli/bench/sweep/serve_workload.md`
- `cli/bench/throughput.md`
- `cli/chat.md`
- `cli/complete.md`
- `cli/json_tip.inc.md`
- `cli/launch/render.md`
- `cli/run-batch.md`
- `cli/serve.md`
- `community/contact_us.md`
- `community/meetups.md`
- `community/sponsors.md`
- `configuration/README.md`
- `configuration/conserving_memory.md`
- `configuration/engine_args.md`
- `configuration/env_vars.md`
- `configuration/model_resolution.md`
- `configuration/optimization.md`
- `configuration/serve_args.md`
- `contributing/ci/failures.md`
- `contributing/ci/nightly_builds.md`
- `contributing/ci/update_pytorch_version.md`
- `contributing/deprecation_policy.md`
- `contributing/dockerfile/dockerfile.md`
- `contributing/editing-agent-instructions.md`
- `contributing/incremental_build.md`
- `contributing/model/README.md`
- `contributing/model/registration.md`
- `contributing/model/tests.md`
- `contributing/vulnerability_management.md`
- `deployment/docker.md`
- `deployment/frameworks/anyscale.md`
- `deployment/frameworks/anything-llm.md`
- `deployment/frameworks/autogen.md`
- `deployment/frameworks/bentoml.md`
- `deployment/frameworks/cerebrium.md`
- `deployment/frameworks/chatbox.md`
- `deployment/frameworks/dify.md`
- `deployment/frameworks/dstack.md`
- `deployment/frameworks/haystack.md`
- `deployment/frameworks/helm.md`
- `deployment/frameworks/hf_inference_endpoints.md`
- `deployment/frameworks/litellm.md`
- `deployment/frameworks/lobe-chat.md`
- `deployment/frameworks/lws.md`
- `deployment/frameworks/modal.md`
- `deployment/frameworks/open-webui.md`
- `deployment/frameworks/retrieval_augmented_generation.md`
- `deployment/frameworks/runpod.md`
- `deployment/frameworks/skypilot.md`
- `deployment/frameworks/streamlit.md`
- `deployment/frameworks/triton.md`
- `deployment/integrations/aibrix.md`
- `deployment/integrations/dynamo.md`
- `deployment/integrations/kaito.md`
- `deployment/integrations/kserve.md`
- `deployment/integrations/kthena.md`
- `deployment/integrations/kubeai.md`
- `deployment/integrations/kuberay.md`
- `deployment/integrations/llamastack.md`
- `deployment/integrations/llm-d.md`
- `deployment/integrations/llmaz.md`
- `deployment/integrations/production-stack.md`
- `deployment/k8s.md`
- `deployment/nginx.md`
- `design/dbo.md`
- `design/endpoint_plugins.md`
- `design/huggingface_integration.md`
- `design/io_processor_plugins.md`
- `design/lora_resolver_plugins.md`
- `design/mm_processing.md`
- `design/optimization_levels.md`
- `design/torch_compile_multimodal.md`
- `features/README.md`
- `features/automatic_prefix_caching.md`
- `features/batch_invariance.md`
- `features/context_extension.md`
- `features/custom_arguments.md`
- `features/disagg_encoder.md`
- `features/disagg_prefill.md`
- `features/index_cache.md`
- `features/interleaved_thinking.md`
- `features/mooncake_connector_usage.md`
- `features/nixl_connector_compatibility.md`
- `features/per_request_metrics.md`
- `features/prompt_embeds.md`
- `features/quantization/README.md`
- `features/quantization/auto_awq.md`
- `features/quantization/bnb.md`
- `features/quantization/fp8_vit_attn.md`
- `features/quantization/gguf.md`
- `features/quantization/gptqmodel.md`
- `features/quantization/inc.md`
- `features/quantization/llm_compressor/README.md`
- `features/quantization/llm_compressor/fp8.md`
- `features/quantization/llm_compressor/int4.md`
- `features/quantization/llm_compressor/int8_w4a8.md`
- `features/quantization/llm_compressor/int8_w8a8.md`
- `features/quantization/modelopt.md`
- `features/quantization/online.md`
- `features/quantization/quantized_kvcache.md`
- `features/quantization/torchao.md`
- `features/sleep_mode.md`
- `features/speculative_decoding/draft_model.md`
- `features/speculative_decoding/dynamic_speculative_decoding.md`
- `features/speculative_decoding/eagle.md`
- `features/speculative_decoding/mlp.md`
- `features/speculative_decoding/mtp.md`
- `features/speculative_decoding/n_gram.md`
- `features/speculative_decoding/parallel_draft_model.md`
- `features/speculative_decoding/speculators.md`
- `features/speculative_decoding/suffix.md`
- `features/structured_outputs.md`
- `getting_started/installation/README.md`
- `getting_started/installation/gpu.cuda.inc.md`
- `getting_started/installation/gpu.md`
- `getting_started/installation/python_env_setup.inc.md`
- `getting_started/quickstart.md`
- `governance/collaboration.md`
- `governance/committers.md`
- `governance/process.md`
- `models/extensions/fastsafetensor.md`
- `models/extensions/instanttensor.md`
- `models/extensions/runai_model_streamer.md`
- `models/extensions/tensorizer.md`
- `models/generative_models.md`
- `models/hardware_supported_models/cpu.md`
- `models/hardware_supported_models/xpu.md`
- `models/pooling_models/reward.md`
- `models/pooling_models/token_classify.md`
- `models/pooling_models/token_embed.md`
- `serving/context_parallel_deployment.md`
- `serving/data_parallel_deployment.md`
- `serving/distributed_troubleshooting.md`
- `serving/integrations/claude_code.md`
- `serving/integrations/codex.md`
- `serving/integrations/langchain.md`
- `serving/integrations/llamaindex.md`
- `serving/offline_inference.md`
- `serving/online_serving/README.md`
- `serving/online_serving/derenderer.md`
- `serving/online_serving/generative_scoring.md`
- `serving/online_serving/openai_compatible_server.md`
- `serving/online_serving/renderer.md`
- `serving/online_serving/speech_to_text.md`
- `serving/parallelism_scaling.md`
- `training/async_rl.md`
- `training/layerwise.md`
- `training/rlhf.md`
- `training/trl.md`
- `training/weight_transfer/README.md`
- `training/weight_transfer/base.md`
- `training/weight_transfer/ipc.md`
- `training/weight_transfer/nccl.md`
- `usage/README.md`
- `usage/faq.md`
- `usage/metrics.md`
- `usage/reproducibility.md`
- `usage/security.md`
- `usage/troubleshooting.md`
- `usage/usage_stats.md`
- `usage/v1_guide.md`
