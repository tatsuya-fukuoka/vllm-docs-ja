# モデルの解決 { #model-resolution }

vLLM は、モデルリポジトリの `config.json` にある `architectures` フィールドを調べ、
vLLM に登録された対応する実装を見つけることで HuggingFace 互換のモデルを読み込みます。
ただし、次の理由でモデルの解決に失敗することがあります。

- モデルリポジトリの `config.json` に `architectures` フィールドがない。
- 非公式のリポジトリが、vLLM に登録されていない別名でモデルを参照している。
- 同じアーキテクチャ名が複数のモデルで使われており、どのモデルを読み込むべきか曖昧である。

これを解消するには、`hf_overrides` オプションで `config.json` の値を上書きし、モデルのアーキテクチャを明示的に指定します。
例:

```python
from vllm import LLM

llm = LLM(
    model="cerebras/Cerebras-GPT-1.3B",
    hf_overrides={"architectures": ["GPT2LMHeadModel"]},  # GPT-2
)
```

vLLM が認識するモデルアーキテクチャは[対応モデルの一覧](../models/supported_models.md)に記載されています。
