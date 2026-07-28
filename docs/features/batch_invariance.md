# バッチ不変性 { #batch-invariance }

!!! note
    バッチ不変性は現在ベータ版です。一部の機能は現在も活発に開発が進められています。
    進捗と今後の改善予定は <https://github.com/vllm-project/vllm/issues/27433> で追えます。

このドキュメントでは、vLLM でバッチ不変性（batch invariance）を有効にする方法を説明します。バッチ不変性とは、モデルの出力が決定的であり、バッチサイズやバッチ内のリクエストの順序に依存しないことを保証する性質です。

## 動機 { #motivation }

バッチ不変性は、いくつかのユースケースで重要になります。

- **フレームワークのデバッグ**: 出力が決定的であれば、バッチングにかかわらず同じ入力が常に同じ出力を生むため、推論フレームワークの問題をデバッグしやすくなります。
- **モデルのデバッグ**: バッチ構成が変わっても挙動が一貫することで、モデル実装の問題を特定しやすくなります。
- **強化学習（RL）**: RL の学習では、再現性と安定した学習のために決定的なロールアウトが必要になることがよくあります。
- **大規模推論システム**: vLLM をコンポーネントとして使うシステムでは、テスト・検証・一貫性の保証の面で決定的な挙動が役立ちます。

## ハードウェア要件 { #hardware-requirements }

バッチ不変性には、compute capability 8.0 以上の NVIDIA GPU が必要です。

## バッチ不変性を有効にする { #enabling-batch-invariance }

バッチ不変性は、環境変数 `VLLM_BATCH_INVARIANT` を `1` に設定することで有効になります。

```bash
export VLLM_BATCH_INVARIANT=1
```

### オンライン推論（サーバーモード） { #online-inference-server-mode }

バッチ不変性を有効にして vLLM サーバーを起動するには次のようにします。

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

そのうえで、OpenAI 互換クライアントを使います。

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### オフライン推論 { #offline-inference }

バッチ不変性を有効にしたオフラインのバッチ推論は次のようになります。

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## 検証済みモデル { #tested-models }

バッチ不変性は、次のモデルでテストおよび検証されています。

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ`
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`
- **Llama 3**: Llama3.1 および 3.2 シリーズ（例: `meta-llama/Llama-3.2-3B-Instruct`）
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`
- **Phi シリーズ**: `microsoft/Phi-3.5-mini-instruct`

他のモデルでも動作する可能性はありますが、明示的に検証されているのは上記のモデルです。特定のモデルで問題が発生した場合は、[GitHub の issue トラッカー](https://github.com/vllm-project/vllm/issues/new/choose)で報告してください。

## 実装の詳細 { #implementation-details }

バッチ不変性を有効にすると、vLLM は次のように動作します。

1. Attention をはじめとする演算に、決定的なカーネル実装を使う
2. バッチサイズが異なっても数値的な挙動が一貫するようにする
3. 非決定性をもたらす可能性のある一部の最適化（テンソル並列時のカスタム all-reduce など）を無効にする

!!! note
    バッチ不変性を有効にすると、既定の非決定的モードと比べて性能に影響が出る場合があります。
    このトレードオフは、再現性を保証するための意図的なものです。

## 今後の改善 { #future-improvements }

バッチ不変性の機能は活発に開発が進められています。予定されている改善は次のとおりです。

- 対応 GPU アーキテクチャの追加
- 対応モデルの拡大
- 性能の最適化
- テストと検証の拡充

最新の状況の確認やアイデアの提案は、[追跡用 issue](https://github.com/vllm-project/vllm/issues/27433) を参照してください。
