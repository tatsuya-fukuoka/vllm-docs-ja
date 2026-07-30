# プーリングモデル { #pooling-models }

!!! note
    現時点でのプーリングモデルのサポートは、主に利便性のためのものです。Hugging Face Transformers や Sentence Transformers を直接使う場合と比べて、性能上の改善が得られることは保証されません。
    vLLM ではプーリングモデルの最適化を予定しています。ご提案があれば <https://github.com/vllm-project/vllm/issues/21796> にコメントしてください。

## プーリングモデルとは { #what-are-pooling-models }

自然言語処理（NLP）は、主に次の 2 種類のタスクに分けられます。

- 自然言語理解（NLU）
- 自然言語生成（NLG）

vLLM がサポートする生成モデルは、よく知られた大規模言語モデル（LLM）、画像・動画・音声などのマルチモーダル入力を扱うマルチモーダルモデル（VLM）、音声認識（speech-to-text）モデル、ストリーミング入力に対応するリアルタイムモデルなど、さまざまなタスク種別を対象としています。これらに共通するのはテキストを生成できることです。さらに vLLM-Omni は、画像・動画・音声を含むマルチモーダルコンテンツの生成をサポートします。

生成モデルの能力が向上し続けるにつれて、これらのモデルが扱える範囲も広がり続けています。しかし、特定のタスクを効率よくこなすために、専用の小さな言語モデルが依然として必要な応用場面もあります。こうしたモデルには通常、次のような特徴があります。

- コンテンツの生成を必要としない。
- ごく限られた機能を果たせればよく、強い汎化性能、創造性、高度な知能を必要としない。
- きわめて低いレイテンシが求められ、コストに制約のあるハードウェア上で動作することもある。
- テキストのみのモデルは通常 10 億パラメータ未満、マルチモーダルモデルでも一般に 100 億パラメータ未満。

これらのモデルは規模としては比較的小さいものの、今日の最先端の大規模言語モデルと同様、あるいはまったく同じ Transformer アーキテクチャにもとづいています。最近公開されたプーリングモデルの多くは大規模言語モデルからファインチューニングされており、大規模モデルの継続的な改善の恩恵を受けられます。このアーキテクチャの類似性により、vLLM のインフラの多くを再利用できます。互換性があるなら、これらのモデルにも vLLM の最新機能を活用してもらえるよう支援していきたいと考えています。

### チートシート { #cheat-sheet }

下図のとおり、プーリングモデルの主要な要素の関係をまとめた図を用意しました。

![チートシート](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/models/pooling_models/cheat_sheet.svg)

### シーケンス単位のタスクとトークン単位のタスク { #sequence-wise-task-and-token-wise-task }

シーケンス単位のタスクとトークン単位のタスクの主な違いは、出力の粒度にあります。シーケンス単位のタスクは入力シーケンス全体に対して 1 つの結果を出力し、トークン単位のタスクはシーケンス内の各トークンごとに結果を出力します。

多くのプーリングモデルは（シーケンス単位の）タスクとトークン単位のタスクの両方をサポートします。既定のプーリングタスク（たとえばシーケンス単位のタスク）が望むものでない場合は、オフラインでは `PoolerConfig(task=<task>)`、オンラインでは `--pooler-config.task <task>` で手動指定する必要があります。

もちろん、入出力のプロセッサをユーザーがカスタマイズできる「plugin」タスクもあります。詳細は [IO プロセッサプラグイン](../../design/io_processor_plugins.md)を参照してください。

### プーリングタスク { #pooling-tasks }

| プーリングタスク         | 粒度   | 出力                                         |
|-----------------------|---------------|-------------------------------------------------|
| `classify`（注記参照） | シーケンス単位 | シーケンスごとのクラスの確率ベクトル |
| `embed`               | シーケンス単位 | シーケンスごとのベクトル表現        |
| `token_classify`      | トークン単位    | トークンごとのクラスの確率ベクトル    |
| `token_embed`         | トークン単位    | トークンごとのベクトル表現           |

!!! note
    分類タスクの中には、Cross-encoder（リランカー）モデルという特殊な下位分類があります。これは、2 つのプロンプトを入力として受け取り、num_labels が 1 の出力を返す分類モデルの一種です。

### プーリングの種類 { #pooling-types }

![プーリングの種類](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/models/pooling_models/pooling_types.svg)

| プーリングの種類  | 粒度   | 説明                                                                                                                                                                                       |
|----------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CLS` プーリング  | シーケンス単位 | BERT 系（双方向 self-attention）のモデルでは、既定で CLS プーリングが使われます。これは、最初のトークン（[CLS] トークン）に対応する last_hidden_states を出力として取ることを意味します。 |
| `LAST` プーリング | シーケンス単位 | GPT 系（因果 self-attention）のモデルでは、既定で LAST プーリングが使われます。これは、最後のトークンに対応する last_hidden_states を出力として取ることを意味します。                           |
| `MEAN` プーリング | シーケンス単位 | 多くの研究で、入力トークン全体にわたって last_hidden_states を平均するほうが、特定の下流タスクで良い性能を示すことが分かっています。そのため、MEAN プーリングを使うモデルが増えています。          |
| `ALL` プーリング  | トークン単位    | すべての入力トークンについて last_hidden_states を出力します。                                                                                                                                              |
| `STEP` プーリング | トークン単位    | returned_token_ids が返すトークン ID に対応する last_hidden_states を絞り込んで出力します。                                                                                         |

### スコア種別 { #score-types }

![スコア種別](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/models/pooling_models/score_types.svg)

スコアリングモデルは、2 つの入力プロンプト間の類似度スコアを計算するためのものです。`cross-encoder`、`late-interaction`、`bi-encoder` の 3 つのモデル種別（`score_type`）をサポートします。

| プーリングタスク         | 粒度   | 出力                                      | スコア種別        | スコアリング関数         |
|-----------------------|---------------|----------------------------------------------|--------------------|--------------------------|
| `classify`（注記参照） | シーケンス単位 | シーケンスごとのリランカースコア             | `cross-encoder`    | 線形分類器        |
| `embed`               | シーケンス単位 | シーケンスごとのベクトル表現     | `bi-encoder`       | コサイン類似度        |
| `token_classify`      | トークン単位    | トークンごとのクラスの確率ベクトル | N/A                | N/A                      |
| `token_embed`         | トークン単位    | トークンごとのベクトル表現        | `late-interaction` | late interaction（MaxSim） |

!!! note
    分類モデルは、num_labels が 1 の出力を持つ場合にのみスコアリングモデルとして使え、スコアリング API を有効にできます。

### プーリングの用途 { #pooling-usages }

| プーリングの用途              | 説明                                                                                                                                               |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| 分類の用途       | 与えられた入力に最もよく対応する、あらかじめ定義されたカテゴリ・クラス・ラベルを予測します。                                                                  |
| 埋め込みの用途            | 非構造データ（テキスト、画像、音声など）を構造化された数値ベクトル（埋め込み）に変換します。                                                    |
| トークン分類の用途 | トークン単位の分類。                                                                                                                                 |
| トークン埋め込みの用途      | トークン単位の埋め込み。                                                                                                                                      |
| 報酬モデルの用途               | 言語モデルが生成した出力の品質を評価し、人間の選好の代理指標として働きます。                                                  |
| スコアリングの用途              | 2 つの入力間の類似度スコアを計算します。`cross-encoder`、`late-interaction`、`bi-encoder` の 3 つのモデル種別（`score_type`）をサポートします。   |
| プラグインの用途              | 入出力のプロセッサをユーザーがカスタマイズできるようにします。詳細は [IO プロセッサプラグイン](../../design/io_processor_plugins.md)を参照してください。 |

複数のプーリングタスクをサポートするモデル、特定の利用場面を想定したモデル、特殊な入出力に対応するモデルもあります。

詳細は以下のリンクを参照してください。

- [分類の使い方](classify.md)
- [埋め込みの使い方](embed.md)
- [トークン分類の使い方](token_classify.md)
- [トークン埋め込みの使い方](token_embed.md)
- [報酬モデルの使い方](reward.md)
- [スコアリングの使い方](scoring.md)
- [特定モデルの例](specific_models.md)

## オフライン推論 { #offline-inference }

vLLM の各プーリングモデルは、[`Pooler.get_supported_tasks`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/pooler/#vllm.model_executor.layers.pooler.Pooler.get_supported_tasks) に従ってこれらのタスクの 1 つ以上をサポートし、対応する API が有効になります。

### プーリングの用途に対応するオフライン API { #offline-apis-corresponding-to-pooling-usages }

| プーリングの用途              | 専用 API       | `LLM.encode` API のプーリングタスク | スコア種別                | スコアリング関数         |
|-----------------------------|---------------------|-----------------------------------|----------------------------|--------------------------|
| 分類の用途       | `LLM.classify(...)` | `classify`                        | `cross-encoder`（注記参照） | 線形分類器        |
| 埋め込みの用途            | `LLM.embed(...)`    | `embed`                           | `bi-encoder`               | コサイン類似度        |
| トークン分類の用途 | N/A                 | `token_classify`                  | N/A                        | N/A                      |
| トークン埋め込みの用途      | N/A                 | `token_embed`                     | `late-interaction`         | late interaction（MaxSim） |
| 報酬モデルの用途               | N/A                 | `classify` と `token_classify`     | N/A                        | N/A                      |
| スコアリングの用途              | `LLM.score(...)`    | N/A                               | N/A                        | N/A                      |
| プラグインの用途              | N/A                 | `plugin`                          | N/A                        | N/A                      |

!!! note
    分類モデルは、num_labels が 1 の出力を持つ場合にのみスコアリングモデルとして使え、スコアリング API を有効にできます。

### `LLM.classify` { #llmclassify }

[`classify`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.classify) メソッドは、プロンプトごとに確率ベクトルを出力します。
主に[分類モデル](classify.md)向けに設計されています。
`LLM.classify` の詳細は[このページ](classify.md#offline-inference)を参照してください。

### `LLM.embed` { #llmembed }

[`embed`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.embed) メソッドは、プロンプトごとに埋め込みベクトルを出力します。
主に[埋め込みモデル](embed.md)向けに設計されています。
`LLM.embed` の詳細は[このページ](embed.md#offline-inference)を参照してください。

### `LLM.score` { #llmscore }

[`score`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.score) メソッドは、文のペア間の類似度スコアを出力します。
主に[スコアモデル](scoring.md)向けに設計されています。

### `LLM.encode` { #llmencode }

[`encode`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.encode) メソッドは、vLLM のすべてのプーリングモデルで利用できます。

より具体的なメソッドを使うか、`LLM.encode` を使う際にタスクを直接指定してください。[上記の表](#offline-apis-corresponding-to-pooling-usages)を参照してください。

### 例 { #examples }

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

## オンラインサービング { #online-serving }

vLLM のオンラインサーバーは、オフライン API に対応するエンドポイントを提供します。

- `LLM.embed` に対応:
    - [Cohere Embed API](embed.md#cohere-embed-api)（`/v2/embed`）
    - [OpenAI 互換 Embeddings API](embed.md#openai-compatible-embeddings-api)（`/v1/embeddings`）
- `LLM.classify` に対応:
    - [Classification API](classify.md#online-serving)（`/classify`）
- `LLM.score` に対応:
    - [Score API](scoring.md#score-api)（`/score`、`/v1/score`）
    - [Cohere Rerank API](scoring.md#rerank-api)（`/rerank`、`/v1/rerank`、`/v2/rerank`）
- Pooling API（`/pooling`）は `LLM.encode` に相当し、すべての種類のプーリングモデルで利用できます。

以下では Pooling API を説明します。その他の API については上記のリンクを参照してください。

### Pooling API { #pooling-api }

vLLM の Pooling API（`/pooling`）は `LLM.encode` に相当し、すべての種類のプーリングモデルで利用できます。

入力形式は [Embeddings API](embed.md#openai-compatible-embeddings-api) と同じですが、出力データは 1 次元の float のリストに限らず、任意の入れ子リストを含められます。

より具体的な API を使うか、Pooling API を使う際にタスクを直接指定してください。[上記の表](#offline-apis-corresponding-to-pooling-usages)を参照してください。

コード例:

- [オンラインの例](../../../examples/pooling/reward/token_reward_online.py)
- [オフラインの例](../../../examples/pooling/reward/token_reward_offline.py)

### 例 { #examples_1 }

```python
# start a supported embeddings model server with `vllm serve`, e.g.
# vllm serve intfloat/e5-small
import requests

host = "localhost"
port = "8000"
model_name = "intfloat/e5-small"

api_url = f"http://{host}:{port}/pooling"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompt = {"model": model_name, "input": prompts, "task": "embed"}

response = requests.post(api_url, json=prompt)

for output in response.json()["data"]:
    data = output["data"]
    print(f"Data: {data!r} (size={len(data)})")
```

## 設定 { #configuration }

vLLM では、プーリングモデルは [`VllmModelForPooling`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/#vllm.model_executor.models.VllmModelForPooling) インターフェースを実装します。
これらのモデルは、入力の最終的な hidden states を取り出して返すために [`Pooler`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/pooler/#vllm.model_executor.layers.pooler.Pooler) を使います。

### モデルランナー { #model-runner }

`--runner pooling` オプションでモデルをプーリングモードで実行します。

!!! tip
    vLLM は `--runner auto` により適切なモデルランナーを自動検出できるため、ほとんどの場合このオプションを設定する必要はありません。

### モデルの変換 { #model-conversion }

vLLM は `--convert <type>` オプションにより、モデルをさまざまなプーリングタスク向けに適合させられます。

`--runner pooling` が（手動または自動で）設定されているにもかかわらず、モデルが [`VllmModelForPooling`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/#vllm.model_executor.models.VllmModelForPooling) インターフェースを実装していない場合、vLLM は下表のアーキテクチャ名にもとづいてモデルの自動変換を試みます。

| アーキテクチャ                                    | `--convert` | サポートされるプーリングタスク      |
|-------------------------------------------------|-------------|------------------------------|
| `*ForTextEncoding`、`*EmbeddingModel`、`*Model` | `embed`     | `token_embed`、`embed`       |
| `*ForRewardModeling`、`*RewardModel`            | `embed`     | `token_embed`、`embed`       |
| `*For*Classification`、`*ClassificationModel`   | `classify`  | `token_classify`、`classify` |

!!! tip
    `--convert <type>` を明示的に設定して、モデルの変換方法を指定できます。

### Pooler の設定 { #pooler-configuration }

#### 定義済みのモデル { #predefined-models }

モデルが定義する [`Pooler`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/layers/pooler/#vllm.model_executor.layers.pooler.Pooler) が `pooler_config` を受け付ける場合、`--pooler-config` オプションでその属性の一部を上書きできます。

#### 変換されたモデル { #converted-models }

モデルが `--convert`（上記参照）によって変換された場合、各タスクに割り当てられる pooler は既定で次の属性を持ちます。

| タスク       | プーリングの種類 | 正規化 | Softmax |
| ---------- | ------------ | ------------- | ------- |
| `embed`    | `LAST`       | ✅︎            | ❌      |
| `classify` | `LAST`       | ❌            | ✅︎      |

#### 解決の優先順位 { #resolution-precedence }

プーリング方式と `use_activation` はフィールドごとに解決されます。`--pooler-config` で明示的に設定されたフィールドは Sentence Transformers のメタデータより優先され、そのメタデータはモデルアーキテクチャやタスクの既定値より優先されます。設定されていないフィールドは、それぞれ独立にこの連鎖をたどります。

現在の `PoolerConfig` には `normalize` や `activation` のフィールドはありません。`use_activation` は、そのタスク用に構築された正規化または分類の活性化を適用するかどうかを制御します。

| フィールド | 参照元の優先順位 | 上書きの方法 |
| ----- | ----------------- | --------------- |
| プーリング方式（`pooling_type`） | `--pooler-config` > Sentence Transformers の `modules.json` が参照する Pooling モジュール内の真偽値フィールド `pooling_mode_*` > アーキテクチャの既定値（アーキテクチャが上書きしない限り、シーケンスプーリングは `LAST`、トークンプーリングは `ALL`） | `{"pooling_type": "CLS"}` を設定するか、`seq_pooling_type` / `tok_pooling_type` を明示的に設定します。 |
| 埋め込みの正規化（`use_activation`） | `--pooler-config` > Sentence Transformers のモジュール（Normalize モジュールがあれば `true`、なければ `false`） > Sentence Transformers の Pooling モジュールが見つからない場合のプーリングタスクの既定値（`true`） | 正規化していない埋め込みを返すには `{"use_activation": false}` を設定します。 |
| 分類の活性化関数 | Hugging Face の `problem_type` > Sentence Transformers の活性化のメタデータ > ラベル数から選ばれる sigmoid または softmax | この関数は `--pooler-config` では選択できません。代わりに logits を返すには `{"use_activation": false}` を設定します。 |

新しいコンパクトな `pooling_mode` 文字列を使う Sentence Transformers の設定は、現時点では解析されません。[issue #45995](https://github.com/vllm-project/vllm/issues/45995) を参照してください。

標準の DispatchPooler アダプタを使う変換済みモデルおよび定義済みモデルでは、`embed` と `token_embed` は L2 正規化のヘッドを構築し、`classify` と `token_classify` は選択された分類の活性化を構築します。どちらの場合も、そのヘッドを適用するかどうかは `use_activation` が制御します。カスタムの pooler を持つモデルは、異なる挙動を実装できます。

モデルの重みを読み込まずに解決後のフィールドを確認するには、次のようにします。

```python
from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.layers.pooler.activations import get_act_fn


def inspect(requested: PoolerConfig) -> None:
    model_config = ModelConfig(
        "intfloat/e5-small",
        runner="pooling",
        pooler_config=requested,
    )
    resolved = model_config.pooler_config
    assert resolved is not None
    print(
        {
            "seq_pooling_type": resolved.seq_pooling_type,
            "tok_pooling_type": resolved.tok_pooling_type,
            "use_activation": resolved.use_activation,
            "sequence_classification_activation": type(
                get_act_fn(model_config.hf_config)
            ).__name__,
        }
    )


inspect(PoolerConfig())
inspect(PoolerConfig(pooling_type="CLS", use_activation=False))
```

`intfloat/e5-small` の場合、1 つ目の結果には `MEAN`、`ALL`、`True` が含まれます。2 つ目には `CLS`、`ALL`、`False` が含まれます。どちらも、標準のシーケンス分類アダプタが構築する分類の活性化を報告します。

## 削除された機能 { #removed-features }

### encode タスク { #encode-task }

`encode` タスクは、より具体的な 2 つのトークン単位のタスク `token_embed` と `token_classify` に分割されました。

- `token_embed` は `embed` と同じで、活性化として正規化を使います。
- `token_classify` は `classify` と同じで、既定では活性化として softmax を使います。

プーリングモデルはトークン単位のタスクをサポートするようになりました。

- hidden states の抽出には `token_embed` タスクの利用が推奨されます。
- 固有表現抽出（NER）や報酬モデルには `token_classify` タスクの利用が推奨されます。

### score タスク { #score-task }

`score` タスクは v0.21 で削除されました。代わりに `classify` を使ってください。分類モデルは、num_labels が 1 の出力を持つ場合にのみスコアリングモデルとして使え、スコアリング API を有効にできます。

### プーリングのマルチタスク対応 { #pooling-multitask-support }

プーリングのマルチタスク対応は v0.21 で削除されました。既定のプーリングタスクが望むものでない場合は、オフラインでは `PoolerConfig(task=<task>)`、オンラインでは `--pooler-config.task <task>` で手動指定する必要があります。
