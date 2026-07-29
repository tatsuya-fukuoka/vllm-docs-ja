# 個別モデルの例 { #specific-model-examples }

## ColBERT 系の late interaction モデル { #colbert-late-interaction-models }

[ColBERT](https://arxiv.org/abs/2004.12832)（Contextualized Late Interaction over BERT）は、トークン単位の埋め込みと MaxSim スコアリングによって文書をランキングする検索モデルです。単一ベクトルの埋め込みモデルとは異なり、ColBERT はトークンレベルの表現を保持し、late interaction によって関連度スコアを計算します。これにより、クロスエンコーダより効率的でありながら高い精度が得られます。

vLLM は、複数のエンコーダバックボーンを持つ ColBERT モデルをサポートしています。

| アーキテクチャ | バックボーン | HF モデルの例 |
| - | - | - |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` |
| `ColBERTLfm2Model` | LFM2 | `LiquidAI/LFM2-ColBERT-350M` |

**BERT ベースの ColBERT** モデルはそのまま動作します。

```shell
vllm serve answerdotai/answerai-colbert-small-v1
```

**BERT 以外のバックボーン**では、`--hf-overrides` で正しいアーキテクチャを指定します。

```shell
# ModernBERT backbone
vllm serve lightonai/GTE-ModernColBERT-v1 \
    --hf-overrides '{"architectures": ["ColBERTModernBertModel"]}'

# Jina XLM-RoBERTa backbone
vllm serve jinaai/jina-colbert-v2 \
    --hf-overrides '{"architectures": ["ColBERTJinaRobertaModel"]}' \
    --trust-remote-code

# LFM2 backbone
vllm serve LiquidAI/LFM2-ColBERT-350M \
    --hf-overrides '{"architectures": ["ColBERTLfm2Model"]}'
```

そのあと、rerank API を使えます。

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

score API を使う場合は次のとおりです。

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "text_1": "What is machine learning?",
    "text_2": ["Machine learning is a subset of AI.", "The weather is sunny."]
}'
```

プーリング API を `token_embed` タスクで使えば、生のトークン埋め込みも取得できます。

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "input": "What is machine learning?",
    "task": "token_embed"
}'
```

例はこちらにあります: [examples/pooling/score/colbert_rerank_online.py](../../../examples/pooling/score/colbert_rerank_online.py)

## ColQwen3 のマルチモーダル late interaction モデル { #colqwen3-multi-modal-late-interaction-models }

ColQwen3 は [ColPali](https://arxiv.org/abs/2407.01449) をベースにしており、ColBERT の late interaction のアプローチを**マルチモーダル**入力に拡張したものです。ColBERT がテキストのみのトークン埋め込みを扱うのに対し、ColPali / ColQwen3 は**テキストと画像**（PDF のページ、スクリーンショット、図など）の両方をトークン単位の L2 正規化ベクトルに埋め込み、MaxSim スコアリングで関連度を計算できます。ColQwen3 は特に Qwen3-VL を vision-language のバックボーンとして使います。

| アーキテクチャ | バックボーン | HF モデルの例 |
| - | - | - |
| `ColQwen3` | Qwen3-VL | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` |
| `OpsColQwen3Model` | Qwen3-VL | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` |

サーバーを起動します。

```shell
vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 4096
```

### テキストのみのスコアリングとリランキング { #text-only-scoring-and-reranking }

`/rerank` API を使います。

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

`/score` API を使う場合は次のとおりです。

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "text_1": "What is the capital of France?",
    "text_2": ["The capital of France is Paris.", "Python is a programming language."]
}'
```

### マルチモーダルのスコアリングとリランキング（テキストのクエリ × 画像の文書） { #multi-modal-scoring-and-reranking-text-query-image-documents }

`/score` と `/rerank` の API は、マルチモーダル入力も直接受け付けます。画像の文書は、`/score` では `data_1` / `data_2`、`/rerank` では `documents` のフィールドに、`image_url` と `text` のパートを含む `content` のリストとして渡します。これは OpenAI の chat completion API と同じ形式です。

テキストのクエリを画像の文書に対してスコアリングします。

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "data_1": "Retrieve the city of Beijing",
    "data_2": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

テキストのクエリで画像の文書をリランキングします。

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "query": "Retrieve the city of Beijing",
    "documents": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_1>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        },
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_2>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ],
    "top_n": 2
}'
```

### 生のトークン埋め込み { #raw-token-embeddings }

`/pooling` API を `token_embed` タスクで使えば、生のトークン埋め込みも取得できます。

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "input": "What is machine learning?",
    "task": "token_embed"
}'
```

プーリング API で**画像を入力**する場合は、チャット形式の `messages` フィールドを使います。

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

### 例 { #examples }

- マルチベクトル検索: [examples/pooling/token_embed/colqwen3_token_embed_online.py](../../../examples/pooling/token_embed/colqwen3_token_embed_online.py)
- リランキング（テキスト + マルチモーダル）: [examples/pooling/score/colqwen3_rerank_online.py](../../../examples/pooling/score/colqwen3_rerank_online.py)

## ColQwen3.5 のマルチモーダル late interaction モデル { #colqwen35-multi-modal-late-interaction-models }

ColQwen3.5 は [ColPali](https://arxiv.org/abs/2407.01449) をベースにしており、ColBERT の late interaction のアプローチを**マルチモーダル**入力に拡張したものです。Qwen3.5 のハイブリッドバックボーン（線形 + full attention）を使い、MaxSim スコアリング用にトークン単位の L2 正規化ベクトルを生成します。

| アーキテクチャ | バックボーン | HF モデルの例 |
| - | - | - |
| `ColQwen3_5` | Qwen3.5 | `athrael-soju/colqwen3.5-4.5B` |

サーバーを起動します。

```shell
vllm serve athrael-soju/colqwen3.5-4.5B --max-model-len 4096
```

そのあと、rerank エンドポイントを使えます。

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "athrael-soju/colqwen3.5-4.5B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

score エンドポイントを使う場合は次のとおりです。

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "athrael-soju/colqwen3.5-4.5B",
    "text_1": "What is the capital of France?",
    "text_2": ["The capital of France is Paris.", "Python is a programming language."]
}'
```

例はこちらにあります: [examples/pooling/score/colqwen3_5_rerank_online.py](../../../examples/pooling/score/colqwen3_5_rerank_online.py)

## Llama Nemotron のマルチモーダル { #llama-nemotron-multimodal }

### 埋め込みモデル { #embedding-model }

Llama Nemotron VL の埋め込みモデルは、双方向の Llama 埋め込みバックボーン（`nvidia/llama-nemotron-embed-1b-v2` 由来）と、ビジョンエンコーダとしての SigLIP を組み合わせ、テキストや画像から単一ベクトルの埋め込みを生成します。

| アーキテクチャ | バックボーン | HF モデルの例 |
| - | - | - |
| `LlamaNemotronVLModel` | 双方向 Llama + SigLIP | `nvidia/llama-nemotron-embed-vl-1b-v2` |

サーバーを起動します。

```shell
vllm serve nvidia/llama-nemotron-embed-vl-1b-v2 \
    --trust-remote-code \
    --chat-template examples/pooling/embed/template/nemotron_embed_vl.jinja
```

!!! note
    このモデルのトークナイザーに同梱されているチャットテンプレートは、埋め込み API には適していません。
    `messages` ベース（チャット形式）の埋め込み API でサービングする場合は、上記の上書き用
    テンプレートを使ってください。

    この上書きテンプレートは、メッセージの `role` に応じて適切な接頭辞を自動的に付加します。
    クエリには `role` を `"query"` に設定し（`query: ` が付加されます）、パッセージには
    `"document"` を設定します（`passage: ` が付加されます）。それ以外の role では接頭辞は付きません。

テキストのクエリを埋め込みます。

```shell
curl -s http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "messages": [
        {
            "role": "query",
            "content": [
                {"type": "text", "text": "What is machine learning?"}
            ]
        }
    ]
}'
```

チャット形式の `messages` フィールドを使って画像を埋め込みます。

```shell
curl -s http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "messages": [
        {
            "role": "document",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

### リランカーモデル { #reranker-model }

Llama Nemotron VL のリランカーモデルは、同じ双方向 Llama + SigLIP のバックボーンに、クロスエンコーダによるスコアリングとリランキングのためのシーケンス分類ヘッドを組み合わせたものです。

| アーキテクチャ | バックボーン | HF モデルの例 |
| - | - | - |
| `LlamaNemotronVLForSequenceClassification` | 双方向 Llama + SigLIP | `nvidia/llama-nemotron-rerank-vl-1b-v2` |

サーバーを起動します。

```shell
vllm serve nvidia/llama-nemotron-rerank-vl-1b-v2 \
    --runner pooling \
    --trust-remote-code \
    --chat-template examples/pooling/score/template/nemotron-vl-rerank.jinja
```

!!! note
    このチェックポイントのトークナイザーに同梱されているチャットテンプレートは、Score / Rerank API には
    適していません。サービング時には、提供されている上書き用テンプレート
    `examples/pooling/score/template/nemotron-vl-rerank.jinja` を使ってください。

テキストのクエリを画像の文書に対してスコアリングします。

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "data_1": "Find diagrams about autonomous robots",
    "data_2": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Robotics workflow diagram."}
            ]
        }
    ]
}'
```

テキストのクエリで画像の文書をリランキングします。

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "query": "Find diagrams about autonomous robots",
    "documents": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_1>"}},
                {"type": "text", "text": "Robotics workflow diagram."}
            ]
        },
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_2>"}},
                {"type": "text", "text": "General skyline photo."}
            ]
        }
    ],
    "top_n": 2
}'
```

## BAAI/bge-m3 { #baaibge-m3 }

`BAAI/bge-m3` モデルには、スパース埋め込みと ColBERT 埋め込みのための追加の重みが含まれています。しかし残念ながら `config.json` ではアーキテクチャが `XLMRobertaModel` と宣言されているため、`vLLM` は追加の重みなしで素の RoBERTa モデルとして読み込んでしまいます。モデルの重みをすべて読み込むには、次のようにアーキテクチャを上書きしてください。

```shell
vllm serve BAAI/bge-m3 --hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}'
```

そのあと、次のようにしてスパース埋め込みを取得できます。

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
     "model": "BAAI/bge-m3",
     "task": "token_classify",
     "input": ["What is BGE M3?", "Definition of BM25"]
}'
```

出力スキーマの制約により、出力は入力ごと・トークンごとのトークンスコアのリストになります。そのため、トークンとスコアを対応づけるには `/tokenize` も呼び出す必要があります。その方法は `tests/models/language/pooling/test_bge_m3.py` のテストを参照してください。

ColBERT 埋め込みは次のようにして取得できます。

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
     "model": "BAAI/bge-m3",
     "task": "token_embed",
     "input": ["What is BGE M3?", "Definition of BM25"]
}'
```
