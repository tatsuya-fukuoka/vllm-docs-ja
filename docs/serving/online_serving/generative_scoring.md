# Generative Scoring { #generative-scoring }

`/generative_scoring` エンドポイントは、CausalLM のモデル（Llama、Qwen、Mistral など）を使って、指定したトークン ID が次のトークンとして現れる確率を計算します。各アイテム（ドキュメント）をクエリと連結してプロンプトを作り、そのプロンプトの次のトークンとして各ラベルのトークンがどれくらい現れやすいかをモデルが予測します。これにより、クエリに対してアイテムをスコアリングできます。たとえば「これはフランスの首都ですか？」と尋ね、モデルが「Yes」と答える確率で各都市をスコアリングする、といった使い方ができます。

このエンドポイントは、生成モデル（タスク `"generate"`）でサーバーを起動すると自動的に利用できます。cross-encoder、bi-encoder、late-interaction のモデルを使うプーリングベースの [Score API](../../models/pooling_models/scoring.md#score-api) とは別のものです。

**要件:**

- `label_token_ids` パラメータは**必須**で、**最低 1 つ**のトークン ID を含む必要があります。
- ラベルのトークンを 2 つ指定した場合、スコアは `P(label_token_ids[0]) / (P(label_token_ids[0]) + P(label_token_ids[1]))` になります（2 つのラベルに対する softmax）。
- 3 つ以上のラベルを指定した場合、スコアはすべてのラベルトークンにわたって softmax で正規化した、最初のラベルトークンの確率になります。

## 仕組み { #how-it-works }

1. **プロンプトの構築**: 各アイテムについて `prompt = query + item` を作ります（`item_first=true` の場合は `item + query`）
2. **順伝播**: 各プロンプトでモデルを実行し、次トークンの logits を得ます
3. **確率の抽出**: 指定された `label_token_ids` の logprobs を取り出します
4. **Softmax による正規化**: ラベルのトークンのみに対して softmax を適用します（`apply_softmax=true` の場合）
5. **スコア**: 最初のラベルトークンの正規化後の確率を返します

## トークン ID の調べ方 { #finding-token-ids }

ラベルに対応するトークン ID は、トークナイザーで調べられます。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
no_id = tokenizer.encode("No", add_special_tokens=False)[0]
print(f"Yes: {yes_id}, No: {no_id}")
```

## 例 { #example }

```bash
curl -X POST http://localhost:8000/generative_scoring \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "query": "Is this city the capital of France?",
    "items": ["Paris", "London", "Berlin"],
    "label_token_ids": [9454, 2753]
  }'
```

この例では、各アイテムがクエリに連結され、`"Is this city the capital of France? Paris"`、`"... London"` のようなプロンプトになります。モデルは次のトークンを予測し、スコアは「Yes」（トークン 9454）と「No」（トークン 2753）の確率の比を表します。

??? console "Response"

    ```json
    {
      "id": "generative-scoring-abc123",
      "object": "list",
      "created": 1234567890,
      "model": "Qwen/Qwen3-0.6B",
      "data": [
        {"index": 0, "object": "score", "score": 0.95},
        {"index": 1, "object": "score", "score": 0.12},
        {"index": 2, "object": "score", "score": 0.08}
      ],
      "usage": {"prompt_tokens": 45, "total_tokens": 48, "completion_tokens": 3}
    }
    ```
