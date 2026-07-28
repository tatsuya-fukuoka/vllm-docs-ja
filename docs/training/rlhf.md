# 人間のフィードバックによる強化学習 (RLHF) { #reinforcement-learning-from-human-feedback }

人間のフィードバックによる強化学習 (RLHF) は、人が作成した嗜好データを使って言語モデルをファインチューニングし、モデルの出力を望ましい振る舞いに合わせる手法です。vLLM は RLHF における補完の生成に利用できます。

次のオープンソースの RL ライブラリは、高速なロールアウトのために vLLM を利用しています（アルファベット順・網羅的ではありません）。

- [Cosmos-RL](https://github.com/nvidia-cosmos/cosmos-rl)
- [ms-swift](https://github.com/modelscope/ms-swift/tree/main)
- [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- [Open Instruct](https://github.com/allenai/open-instruct)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PipelineRL](https://github.com/ServiceNow/PipelineRL)
- [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [TRL](https://github.com/huggingface/trl)
- [Unsloth](https://github.com/unslothai/unsloth)
- [verl](https://github.com/volcengine/verl)

学習側と推論側の重みの同期については[重み転送](weight_transfer/README.md)のドキュメントを参照してください。[NCCL](weight_transfer/nccl.md)（複数 GPU）と [IPC](weight_transfer/ipc.md)（同一 GPU）のエンジンを差し替えられる仕組みを説明しています。

生成と学習をパイプライン化して GPU の使用率とスループットを高める方法は[非同期強化学習](async_rl.md)のガイドを参照してください。実行中に安全に重みを更新するための pause / resume API を説明しています。

vLLM を GRPO で使う方法は、次のノートブックを参照してください。

- [Efficient Online Training with GRPO and vLLM in TRL](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)（英語）
- [Qwen-3 4B GRPO using Unsloth + vLLM](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)（英語）
