# Transformers Reinforcement Learning (TRL) { #transformers-reinforcement-learning }

[Transformers Reinforcement Learning](https://huggingface.co/docs/trl) (TRL) は、教師ありファインチューニング (SFT)、Group Relative Policy Optimization (GRPO)、Direct Preference Optimization (DPO)、報酬モデリングなどの手法で Transformer 系の言語モデルを学習するためのツール群を提供するフルスタックのライブラリです。🤗 transformers と統合されています。

GRPO や Online DPO のようなオンライン手法では、モデルが補完を生成する必要があります。この生成に vLLM を利用できます。

詳細は TRL のドキュメントにある [vLLM 統合ガイド](https://huggingface.co/docs/trl/main/en/vllm_integration)（英語）を参照してください。

TRL は現在、vLLM と組み合わせて次のオンライントレーナーをサポートしています。

- [GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Online DPO](https://huggingface.co/docs/trl/main/en/online_dpo_trainer)
- [RLOO](https://huggingface.co/docs/trl/main/en/rloo_trainer)
- [Nash-MD](https://huggingface.co/docs/trl/main/en/nash_md_trainer)
- [XPO](https://huggingface.co/docs/trl/main/en/xpo_trainer)

TRL で vLLM を有効にするには、トレーナーの設定で `use_vllm` フラグを `True` にします。

## 学習時の vLLM の使い方（モード） { #modes-of-using-vllm-during-training }

TRL は学習時の vLLM の統合について **2 つのモード**をサポートしています。**サーバーモード**と**コロケートモード**です。`vllm_mode` パラメータで切り替えます。

### サーバーモード { #server-mode }

**サーバーモード**では、vLLM は専用の GPU 上で独立したプロセスとして動作し、HTTP リクエストでトレーナーと通信します。推論用に別の GPU を用意できる場合に適した構成で、生成のワークロードを学習から分離できるため、性能が安定しスケールも容易になります。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="server",  # default value, can be omitted
)
```

### コロケートモード { #colocate-mode }

**コロケートモード**では、vLLM はトレーナーのプロセス内で動作し、学習中のモデルと GPU メモリを共有します。別途サーバーを起動する必要がなく GPU の使用効率を高められますが、学習用 GPU でメモリの競合が起きる可能性があります。

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

一部のトレーナーは **vLLM のスリープモード**にも対応しており、学習中にパラメータとキャッシュを GPU の RAM へ退避してメモリ使用量を抑えられます。詳細は[メモリ最適化のドキュメント](https://huggingface.co/docs/trl/main/en/reducing_memory_usage#vllm-sleep-mode)（英語）を参照してください。

!!! info
    詳細な設定項目やフラグについては、使用するトレーナーのドキュメントを参照してください。
