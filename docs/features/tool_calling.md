# ツール呼び出し { #tool-calling }

vLLM は現在、chat completion API の `tool_choice` フィールドについて、名前付き関数呼び出しに加え、`auto`、`required`（`vllm>=0.8.3` 以降）、`none` のオプションをサポートしています。

## クイックスタート { #quickstart }

ツール呼び出しを有効にしてサーバーを起動します。この例では Meta の Llama 3.1 8B モデルを使うため、vLLM の examples ディレクトリにある `llama3_json` 用のツール呼び出しチャットテンプレートを指定します。

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
```

次に、モデルが利用可能なツールを使うようなリクエストを送ります。

??? code

    ```python
    from openai import OpenAI
    import json

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    def get_weather(location: str, unit: str):
        return f"Getting the weather for {location} in {unit}..."
    tool_functions = {"get_weather": get_weather}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state, e.g., 'San Francisco, CA'"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location", "unit"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
    )

    tool_call = response.choices[0].message.tool_calls[0].function
    print(f"Function called: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")
    print(f"Result: {tool_functions[tool_call.name](**json.loads(tool_call.arguments))}")
    ```

出力例:

```text
Function called: get_weather
Arguments: {"location": "San Francisco, CA", "unit": "fahrenheit"}
Result: Getting the weather for San Francisco, CA in fahrenheit...
```

この例では次を示しています。

* ツール呼び出しを有効にしたサーバーのセットアップ
* ツール呼び出しを処理する実際の関数の定義
* `tool_choice="auto"` を指定したリクエストの送信
* 構造化されたレスポンスの処理と、対応する関数の実行

`tool_choice={"type": "function", "function": {"name": "get_weather"}}` を設定すれば、名前付き関数呼び出しで特定の関数を指定することもできます。この場合は構造化出力のバックエンドが使われるため、初回の利用時には FSM のコンパイルに数秒（あるいはそれ以上）のレイテンシが生じます。以降のリクエストではキャッシュされます。

次の点は呼び出し側の責任であることに注意してください。

1. リクエストで適切なツールを定義すること
2. チャットメッセージに関連する文脈を含めること
3. アプリケーションのロジックでツール呼び出しを処理すること

並列のツール呼び出しやモデルごとのパーサーなど、より高度な使い方は以下の各節を参照してください。

## 名前付き関数呼び出し { #named-function-calling }

vLLM は chat completion API の名前付き関数呼び出しを既定でサポートします。これは vLLM がサポートするほとんどの構造化出力バックエンドで動作するはずです。ここで保証されるのは、構文的に正しく解析できる関数呼び出しであり、内容の品質が高いことではありません。

vLLM は構造化出力を使い、レスポンスが `tools` パラメータの JSON スキーマで定義されたツールのパラメータオブジェクトと一致するようにします。
最良の結果を得るには、期待される出力形式 / スキーマをプロンプトにも明示し、モデルが意図する生成内容と、構造化出力バックエンドが強制するスキーマを揃えることを推奨します。

名前付き関数を使うには、chat completion リクエストの `tools` パラメータで関数を定義し、`tool_choice` パラメータでいずれかのツールの `name` を指定します。

## 必須の関数呼び出し（required） { #required-function-calling }

vLLM は chat completion API の `tool_choice='required'` オプションをサポートします。名前付き関数呼び出しと同様に構造化出力を使うため、既定で有効であり、サポートされる任意のモデルで動作します。ただし、代替のデコードバックエンドのサポートは V1 エンジンの[ロードマップ](../usage/v1_guide.md#features)にあります。

tool_choice='required' を設定すると、モデルは `tools` パラメータで指定されたツール一覧にもとづき、1 つ以上のツール呼び出しを必ず生成します。ツール呼び出しの数はユーザーのクエリによって決まります。出力形式は `tools` パラメータで定義したスキーマに厳密に従います。

## ツールを使わない設定（none） { #none-function-calling }

vLLM は chat completion API の `tool_choice='none'` オプションをサポートします。このオプションを設定すると、リクエストでツールが定義されていても、モデルはツール呼び出しを生成せず、通常のテキストのみで応答します。

!!! note
    リクエストでツールが指定されている場合、vLLM は `tool_choice` の設定にかかわらず、既定でプロンプトにツールの定義を含めます。`tool_choice='none'` のときにツール定義を除外するには、`--exclude-tools-when-tool-choice-none` オプションを使ってください。

## 制約付きデコードの挙動 { #constrained-decoding-behavior }

生成時に vLLM がツールのパラメータスキーマを強制するかどうかは、`tool_choice` のモードとツールごとの `strict` フィールドによって決まります。

| `tool_choice` の値 | スキーマ制約付きデコード | 挙動 |
| --- | --- | --- |
| 名前付き関数 | あり（構造化出力バックエンド経由） | 引数は、その関数のパラメータスキーマに準拠した正しい JSON であることが保証されます。 |
| `"required"` | あり（構造化出力バックエンド経由） | 名前付き関数と同じです。モデルは少なくとも 1 つのツール呼び出しを生成する必要があります。 |
| `"auto"` | 少なくとも 1 つのツールに `strict: true` が設定されている場合のみ | ツールが `strict: true` で明示的に選択した場合、構造タグのパーサーがツール呼び出しの引数を制約します。指定がない場合、モデルは自由に生成し、ツール呼び出しは生のテキストから抽出されます。 |
| `"none"` | 該当なし | ツール呼び出しは生成されません。 |

### strict モード { #strict-mode }

`tool_choice="required"` または名前付き関数呼び出しでは、`strict` フィールドにかかわらず、構造タグの制約が常に適用されます。`tool_choice="auto"` では、少なくとも 1 つのツールに `strict: true` を設定することで構造タグの制約が有効になります。指定がない場合、モデルは自由に生成し、ツール呼び出しは生のテキストから抽出されます。`strict` フィールドは、Chat Completion、Responses、Anthropic Messages の 3 つの API すべてでサポートされます。

厳密なスキーマ強制との互換性を高めるには、ツールのパラメータスキーマを OpenAI の strict-schema 形式で定義してください。

* `parameters` 内の各オブジェクトで `additionalProperties` を `false` に設定します。
* `properties` のすべてのフィールドを必須にします。
* 省略可能なフィールドは `null` を許容する形で表現します（例: `{"type": ["string", "null"]}`）。

vLLM は環境変数 `VLLM_ENFORCE_STRICT_TOOL_CALLING`（既定値 `true`）によるグローバルな切り替えも提供します。`false` に設定すると、ツールごとの `strict` フィールドにかかわらず、vLLM はツール呼び出しに構造タグを付与しません。この環境変数が影響するのは構造タグにもとづくツール呼び出しのみで、名前付き関数呼び出しや `tool_choice="required"` で使われるスキーマ由来の構造化出力は変わりません。

```bash
VLLM_ENFORCE_STRICT_TOOL_CALLING=false vllm serve ...
```

## 自動的な関数呼び出し { #automatic-function-calling }

この機能を有効にするには、次のフラグを設定します。

* `--enable-auto-tool-choice` — **必須**。自動的なツール選択です。モデルが適切と判断したときに自ら tool call を生成できるようにすることを vLLM に伝えます。
* `--tool-call-parser` — 使用するツールパーサーを選択します（一覧は下記）。ツールパーサーは今後も追加されていきます。`--tool-parser-plugin` で自作のツールパーサーを登録することもできます。
* `--tool-parser-plugin` — **任意**。ユーザー定義のツールパーサーを vLLM に登録するためのツールパーサープラグインです。登録したツールパーサー名は `--tool-call-parser` で指定できます。
* `--chat-template` — 自動ツール選択では**任意**です。`tool` ロールのメッセージや、過去に生成されたツール呼び出しを含む `assistant` ロールのメッセージを扱うチャットテンプレートへのパスです。Hermes、Mistral、Llama の各モデルは `tokenizer_config.json` にツール対応のチャットテンプレートを持っていますが、独自のテンプレートを指定することもできます。モデルの `tokenizer_config.json` にツール利用専用のチャットテンプレートが設定されている場合、この引数に `tool_use` を指定できます。その場合は `transformers` の仕様に従って使われます。詳細は HuggingFace の[こちら](https://huggingface.co/docs/transformers/en/chat_templating#why-do-some-models-have-multiple-templates)を参照してください。`tokenizer_config.json` の例は[こちら](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/blob/main/tokenizer_config.json)にあります。

お気に入りのツール呼び出し対応モデルがサポートされていない場合は、パーサーとツール利用チャットテンプレートのコントリビューションをぜひご検討ください。

!!! note
    `tool_choice="auto"` では、スキーマレベルの制約に `VLLM_ENFORCE_STRICT_TOOL_CALLING=true`（既定）と、`strict: true` を持つツールが少なくとも 1 つ必要です。これらの条件が満たされ、選択したパーサーが構造タグをサポートしている場合、vLLM はツール呼び出しの引数を制約します。そうでない場合、vLLM は生のテキストからツール呼び出しを抽出するため、引数が不正な形式であったり、関数のパラメータスキーマに違反したりすることがあります。

### Hermes 系モデル（`hermes`） { #hermes-models-hermes }

Hermes 2 Pro より新しい Nous Research の Hermes シリーズのモデルはすべてサポートされているはずです。

* `NousResearch/Hermes-2-Pro-*`
* `NousResearch/Hermes-2-Theta-*`
* `NousResearch/Hermes-3-*`

_なお、Hermes 2 **Theta** のモデルは、作成過程のマージ手順が原因でツール呼び出しの品質と能力が低下していることが知られています_。

フラグ: `--tool-call-parser hermes`

### Mistral 系モデル（`mistral`） { #mistral-models-mistral }

サポートされるモデル:

* `mistralai/Mistral-7B-Instruct-v0.3`（確認済み）
* その他の Mistral の関数呼び出し対応モデルも互換です。

既知の問題:

1. Mistral 7B は並列のツール呼び出しを正しく生成するのが苦手です。
2. **Transformers のトークナイズバックエンドの場合のみ**: Mistral の `tokenizer_config.json` のチャットテンプレートは、ちょうど 9 桁のツール呼び出し ID を必要とし、これは vLLM が生成するものよりずっと短いものです。この条件が満たされないと例外が送出されるため、次の追加のチャットテンプレートが用意されています。

    * [examples/tool_chat_template_mistral.jinja](../../examples/tool_chat_template_mistral.jinja) - 「公式」の Mistral チャットテンプレートを、vLLM のツール呼び出し ID で動くよう調整したものです（`tool_call_id` フィールドは末尾 9 桁に切り詰められます）
    * [examples/tool_chat_template_mistral_parallel.jinja](../../examples/tool_chat_template_mistral_parallel.jinja) - ツールが与えられたときにツール利用のシステムプロンプトを追加する「改良版」で、並列のツール呼び出しの信頼性が大きく向上します。

推奨フラグ:

1. Mistral AI 公式の形式を使う場合:

    `--tool-call-parser mistral`

2. 利用可能な場合に Transformers の形式を使う場合:

    `--tokenizer_mode hf --config_format hf --load_format hf --tool-call-parser mistral --chat-template examples/tool_chat_template_mistral_parallel.jinja`

!!! note
    Mistral AI が公式にリリースしているモデルには 2 つの形式があります。

    1. `auto` または `mistral` の引数で既定で使われる公式の形式:

        `--tokenizer_mode mistral --config_format mistral --load_format mistral`
        この形式は Mistral AI のトークナイザーバックエンドである [mistral-common](https://github.com/mistralai/mistral-common) を使います。

    2. 利用可能な場合に `hf` の引数で使われる Transformers の形式:

        `--tokenizer_mode hf --config_format hf --load_format hf --chat-template examples/tool_chat_template_mistral_parallel.jinja`

### Llama 系モデル（`llama3_json`） { #llama-models-llama3_json }

サポートされるモデル:

Llama 3.1、3.2、4 のすべてのモデルがサポートされているはずです。

* `meta-llama/Llama-3.1-*`
* `meta-llama/Llama-3.2-*`
* `meta-llama/Llama-4-*`

サポートされるのは [JSON ベースのツール呼び出し](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling)です。Llama-3.2 のモデルで導入された [pythonic なツール呼び出し](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md#zero-shot-function-calling)については、後述の `pythonic` ツールパーサーを参照してください。Llama 4 のモデルでは `llama4_pythonic` ツールパーサーの利用を推奨します。

組み込みの python ツール呼び出しやカスタムのツール呼び出しなど、その他の形式はサポートされていません。

既知の問題:

1. Llama 3 では並列のツール呼び出しはサポートされませんが、Llama 4 のモデルではサポートされます。
2. モデルが誤った形式のパラメータを生成することがあります。たとえば配列ではなく、文字列としてシリアライズされた配列を生成する、といったケースです。

vLLM は Llama 3.1 と 3.2 向けに 2 つの JSON ベースのチャットテンプレートを提供しています。

* [examples/tool_chat_template_llama3.1_json.jinja](../../examples/tool_chat_template_llama3.1_json.jinja) - Llama 3.1 モデル向けの「公式」チャットテンプレートを、vLLM でより良く動くよう調整したものです。
* [examples/tool_chat_template_llama3.2_json.jinja](../../examples/tool_chat_template_llama3.2_json.jinja) - Llama 3.1 のチャットテンプレートを拡張し、画像のサポートを追加したものです。

推奨フラグ: `--tool-call-parser llama3_json --chat-template {上記参照}`

vLLM は Llama 4 向けに pythonic と JSON ベースのチャットテンプレートも提供していますが、pythonic なツール呼び出しを推奨します。

* [examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) - Llama 4 モデル向けの[公式チャットテンプレート](https://www.llama.com/docs/model-cards-and-prompt-formats/llama4/)にもとづくものです。

Llama 4 モデルでは `--tool-call-parser llama4_pythonic --chat-template examples/tool_chat_template_llama4_pythonic.jinja` を使ってください。

### IBM Granite { #ibm-granite }

サポートされるモデル:

* `ibm-granite/granite-4.0-h-small` およびその他の Granite 4.0 モデル

    推奨フラグ: `--tool-call-parser granite4`

* `ibm-granite/granite-3.0-8b-instruct`

    推奨フラグ: `--tool-call-parser granite --chat-template examples/tool_chat_template_granite.jinja`

    [examples/tool_chat_template_granite.jinja](../../examples/tool_chat_template_granite.jinja): Hugging Face 上の元のテンプレートを修正したものです。並列の関数呼び出しをサポートします。

* `ibm-granite/granite-3.1-8b-instruct`

    推奨フラグ: `--tool-call-parser granite`

    Hugging Face のチャットテンプレートをそのまま使えます。並列の関数呼び出しをサポートします。

* `ibm-granite/granite-20b-functioncalling`

    推奨フラグ: `--tool-call-parser granite-20b-fc --chat-template examples/tool_chat_template_granite_20b_fc.jinja`

    [examples/tool_chat_template_granite_20b_fc.jinja](../../examples/tool_chat_template_granite_20b_fc.jinja): Hugging Face 上の元のテンプレート（vLLM と互換ではありません）を修正したものです。Hermes テンプレートの関数記述の要素を取り入れ、[論文](https://arxiv.org/abs/2407.00121)の「Response Generation」モードと同じシステムプロンプトに従います。並列の関数呼び出しをサポートします。

### InternLM 系モデル（`internlm`） { #internlm-models-internlm }

サポートされるモデル:

* `internlm/internlm2_5-7b-chat`（確認済み）
* その他の internlm2.5 の関数呼び出し対応モデルも互換です

既知の問題:

* この実装は InternLM2 もサポートしますが、`internlm/internlm2-chat-7b` モデルでのテストではツール呼び出しの結果が安定しません。

推奨フラグ: `--tool-call-parser internlm --chat-template examples/tool_chat_template_internlm2_tool.jinja`

### Jamba 系モデル（`jamba`） { #jamba-models-jamba }

AI21 の Jamba-1.5 モデルがサポートされています。

* `ai21labs/AI21-Jamba-1.5-Mini`
* `ai21labs/AI21-Jamba-1.5-Large`

フラグ: `--tool-call-parser jamba`

### xLAM 系モデル（`xlam`） { #xlam-models-xlam }

xLAM のツールパーサーは、さまざまな JSON 形式でツール呼び出しを生成するモデルをサポートするよう設計されています。次のような複数の出力スタイルの関数呼び出しを検出します。

1. 直接の JSON 配列: `[` で始まり `]` で終わる JSON 配列の出力文字列
2. thinking タグ: JSON 配列を含む `<think>...</think>` タグ
3. コードブロック: コードブロック内の JSON（```json ...```）
4. ツール呼び出しタグ: `[TOOL_CALLS]` または `<tool_call>...</tool_call>` タグ

並列の関数呼び出しをサポートし、テキストの内容とツール呼び出しを適切に分離できます。

サポートされるモデル:

* Salesforce の Llama-xLAM モデル: `Salesforce/Llama-xLAM-2-8B-fc-r`、`Salesforce/Llama-xLAM-2-70B-fc-r`
* Qwen-xLAM モデル: `Salesforce/xLAM-1B-fc-r`、`Salesforce/xLAM-3B-fc-r`、`Salesforce/Qwen-xLAM-32B-fc-r`

フラグ:

* Llama ベースの xLAM モデル: `--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_llama.jinja`
* Qwen ベースの xLAM モデル: `--tool-call-parser xlam --chat-template examples/tool_chat_template_xlam_qwen.jinja`

### Qwen 系モデル { #qwen-models }

Qwen2.5 では、tokenizer_config.json のチャットテンプレートにすでに Hermes 形式のツール利用のサポートが含まれています。したがって、Qwen モデルのツール呼び出しを有効にするには `hermes` パーサーを使えます。詳細は公式の [Qwen ドキュメント](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm)を参照してください。

* `Qwen/Qwen2.5-*`
* `Qwen/QwQ-32B`

フラグ: `--tool-call-parser hermes`

### DeepSeek-V3 系モデル（`deepseek_v3`） { #deepseek-v3-models-deepseek_v3 }

サポートされるモデル:

* `deepseek-ai/DeepSeek-V3-0324`（[examples/tool_chat_template_deepseekv3.jinja](../../examples/tool_chat_template_deepseekv3.jinja) とともに使用）
* `deepseek-ai/DeepSeek-R1-0528`（[examples/tool_chat_template_deepseekr1.jinja](../../examples/tool_chat_template_deepseekr1.jinja) とともに使用）

フラグ: `--tool-call-parser deepseek_v3 --chat-template {上記参照}`

### DeepSeek-V3.1 系モデル（`deepseek_v31`） { #deepseek-v31-models-deepseek_v31 }

サポートされるモデル:

* `deepseek-ai/DeepSeek-V3.1`（[examples/tool_chat_template_deepseekv31.jinja](../../examples/tool_chat_template_deepseekv31.jinja) とともに使用）

フラグ: `--tool-call-parser deepseek_v31 --chat-template {上記参照}`

### OpenAI OSS モデル（`openai`） { #openai-oss-models-openai }

サポートされるモデル:

* `openai/gpt-oss-20b`
* `openai/gpt-oss-120b`

フラグ: `--tool-call-parser openai`

### Kimi-K2 系モデル（`kimi_k2`） { #kimi-k2-models-kimi_k2 }

サポートされるモデル:

* `moonshotai/Kimi-K2-Instruct`

フラグ: `--tool-call-parser kimi_k2`

### Hunyuan 系モデル（`hunyuan_a13b`） { #hunyuan-models-hunyuan_a13b }

サポートされるモデル:

* `tencent/Hunyuan-A13B-Instruct`（チャットテンプレートは Hugging Face のモデルファイルに含まれています）

フラグ:

* 推論なしの場合: `--tool-call-parser hunyuan_a13b`
* 推論ありの場合: `--tool-call-parser hunyuan_a13b --reasoning-parser hunyuan_a13b`

### Cohere Command A Reasoning（`cohere_command3`） { #cohere-command-a-reasoning-cohere_command3 }

サポートされるモデル:

* [`CohereLabs/command-a-reasoning-08-2025`](https://huggingface.co/CohereLabs/command-a-reasoning-08-2025)

フラグ: `--tool-call-parser cohere_command3 --reasoning-parser cohere_command3`

注: Cohere のツールパーサーは `cohere_melody` パッケージを必要としますが、既定ではインストールされません。このパーサーを使う前に [cohere_melody](https://pypi.org/project/cohere-melody/) パッケージをインストールしてください。

### LongCat-Flash-Chat 系モデル（`longcat`） { #longcat-flash-chat-models-longcat }

サポートされるモデル:

* `meituan-longcat/LongCat-Flash-Chat`
* `meituan-longcat/LongCat-Flash-Chat-FP8`

フラグ: `--tool-call-parser longcat`

### GLM-4.5 系モデル（`glm45`） { #glm-45-models-glm45 }

サポートされるモデル:

* `zai-org/GLM-4.5`
* `zai-org/GLM-4.5-Air`
* `zai-org/GLM-4.6`

フラグ: `--tool-call-parser glm45`

### GLM-4.7 系モデル（`glm47`） { #glm-47-models-glm47 }

サポートされるモデル:

* `zai-org/GLM-4.7`
* `zai-org/GLM-4.7-Flash`

フラグ: `--tool-call-parser glm47`

### FunctionGemma 系モデル（`functiongemma`） { #functiongemma-models-functiongemma }

Google の FunctionGemma は、関数呼び出しに特化して設計された軽量（2.7 億パラメータ）のモデルです。
Gemma 3 をベースに構築され、ノート PC やスマートフォンなどのデバイス上でのエッジデプロイ向けに最適化されています。

サポートされるモデル:

* `google/functiongemma-270m-it`

FunctionGemma は `<start_function_call>` と `<end_function_call>` タグを使う独自の出力形式を採用しています。

```text
<start_function_call>call:get_weather{location:<escape>London<escape>}<end_function_call>
```

最良の結果を得るには、特定の関数呼び出しタスク向けにファインチューニングすることを想定した設計になっています。

フラグ: `--tool-call-parser functiongemma --chat-template examples/tool_chat_template_functiongemma.jinja`

!!! note
    FunctionGemma は、あなたの特定の関数呼び出しタスク向けにファインチューニングして使うことが想定されています。
    ベースモデルでも一般的な関数呼び出しの能力は得られますが、最良の結果はタスク固有の
    ファインチューニングによって得られます。ファインチューニングのガイドは Google の [FunctionGemma ドキュメント](https://ai.google.dev/gemma/docs/functiongemma)を参照してください。

### Qwen3-Coder 系モデル（`qwen3_xml`） { #qwen3-coder-models-qwen3_xml }

サポートされるモデル:

* `Qwen/Qwen3-Coder-480B-A35B-Instruct`
* `Qwen/Qwen3-Coder-30B-A3B-Instruct`

フラグ: `--tool-call-parser qwen3_xml`

### Olmo 3 系モデル（`olmo3`） { #olmo-3-models-olmo3 }

Olmo 3 のモデルは、後述の `pythonic` パーサーが想定する形式に非常に近い形式でツール呼び出しを出力しますが、いくつか違いがあります。各ツール呼び出しは pythonic な文字列ですが、並列のツール呼び出しは改行区切りで、呼び出し全体が `<function_calls>..</function_calls>` の XML タグで囲まれます。さらにこのパーサーは、pythonic のリテラル（`True`、`False`、`None`）に加えて JSON の真偽値・null リテラル（`true`、`false`、`null`）も許容します。

サポートされるモデル:

* `allenai/Olmo-3-7B-Instruct`
* `allenai/Olmo-3-32B-Think`

フラグ: `--tool-call-parser olmo3`

### Gigachat 3 系モデル（`gigachat3`） { #gigachat-3-models-gigachat3 }

Hugging Face のモデルファイルにあるチャットテンプレートを使ってください。

サポートされるモデル:

* `ai-sage/GigaChat3-702B-A36B-preview`
* `ai-sage/GigaChat3-702B-A36B-preview-bf16`
* `ai-sage/GigaChat3-10B-A1.8B`
* `ai-sage/GigaChat3-10B-A1.8B-bf16`

フラグ: `--tool-call-parser gigachat3`

### Apertus 系モデル（`apertus`） { #apertus-models-apertus }

examples フォルダのチャットテンプレートを使ってください。OpenAI 互換性の問題がいくつか修正されています: `--chat-template /vllm-workspace/examples/tool_chat_template_apertus.jinja`

サポートされるモデル:

* `swiss-ai/Apertus-8B-Instruct-2509`
* `swiss-ai/Apertus-70B-Instruct-2509`

フラグ: `--tool-call-parser apertus`

### pythonic なツール呼び出しを行うモデル（`pythonic`） { #models-with-pythonic-tool-calls-pythonic }

JSON ではなく python のリストでツール呼び出しを表現するモデルが増えています。この方式には、並列のツール呼び出しを本質的にサポートでき、ツール呼び出しに必要な JSON スキーマの曖昧さを取り除けるという利点があります。`pythonic` ツールパーサーはこうしたモデルをサポートできます。

具体例として、こうしたモデルはサンフランシスコとシアトルの天気を調べるために次のような出力を生成します。

```python
[get_weather(city='San Francisco', metric='celsius'), get_weather(city='Seattle', metric='celsius')]
```

制限事項:

* モデルは、同一の生成の中でテキストとツール呼び出しの両方を生成してはいけません。特定のモデルについてはこれを変えるのは難しくないかもしれませんが、ツール呼び出しの開始と終了にどのトークンを出力すべきかについて、現時点でコミュニティの合意がありません。（とくに Llama 3.2 のモデルはそうしたトークンを出力しません。）
* Llama の小さいモデルは、ツールを効果的に使うのが苦手です。

サポートされるモデルの例:

* `meta-llama/Llama-3.2-1B-Instruct` ⚠️（[examples/tool_chat_template_llama3.2_pythonic.jinja](../../examples/tool_chat_template_llama3.2_pythonic.jinja) とともに使用）
* `meta-llama/Llama-3.2-3B-Instruct` ⚠️（[examples/tool_chat_template_llama3.2_pythonic.jinja](../../examples/tool_chat_template_llama3.2_pythonic.jinja) とともに使用）
* `Team-ACE/ToolACE-8B`（[examples/tool_chat_template_toolace.jinja](../../examples/tool_chat_template_toolace.jinja) とともに使用）
* `fixie-ai/ultravox-v0_4-ToolACE-8B`（[examples/tool_chat_template_toolace.jinja](../../examples/tool_chat_template_toolace.jinja) とともに使用）
* `meta-llama/Llama-4-Scout-17B-16E-Instruct` ⚠️（[examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) とともに使用）
* `meta-llama/Llama-4-Maverick-17B-128E-Instruct` ⚠️（[examples/tool_chat_template_llama4_pythonic.jinja](../../examples/tool_chat_template_llama4_pythonic.jinja) とともに使用）

フラグ: `--tool-call-parser pythonic --chat-template {上記参照}`

!!! warning
    Llama の小さいモデルは、正しい形式でツール呼び出しを出力できないことが頻繁にあります。結果はモデルによって異なります。

## ツール呼び出し性能のベンチマーク { #benchmarking-tool-calling-performance }

現実的なツール呼び出しのトラフィックにおけるサービングのレイテンシとスループットを測定するには、
`vllm bench serve` とともに BFCL（Berkeley Function Calling Leaderboard）データセットを使います。
サーバー側・クライアント側の完全なコマンドは [BFCL ベンチマークの例](../benchmarking/cli.md#bfcl-tool-calling-benchmark)を参照してください。

## ツールパーサープラグインの書き方 { #how-to-write-a-tool-parser-plugin }

ツールパーサープラグインは、1 つ以上の ToolParser 実装を含む Python ファイルです。[vllm/tool_parsers/hermes_tool_parser.py](../../vllm/tool_parsers/hermes_tool_parser.py) の `Hermes2ProToolParser` と同様に ToolParser を書けます。

プラグインファイルの概要は次のとおりです。

??? code

    ```python

    # import the required packages

    # define a tool parser and register it to vllm
    # the name list in register_module can be used
    # in --tool-call-parser. you can define as many
    # tool parsers as you want here.
    class ExampleToolParser(ToolParser):
        def __init__(self, tokenizer: TokenizerLike):
            super().__init__(tokenizer)

        # adjust request. e.g.: set skip special tokens
        # to False for tool call output.
        def adjust_request(self, request: ChatCompletionRequest | ResponsesRequest) -> ChatCompletionRequest | ResponsesRequest:
            return request

        # implement the tool call parse for stream call
        def extract_tool_calls_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
            request: ChatCompletionRequest,
        ) -> DeltaMessage | None:
            return delta

        # implement the tool parse for non-stream call
        def extract_tool_calls(
            self,
            model_output: str,
            request: ChatCompletionRequest,
        ) -> ExtractedToolCallInformation:
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=text)
    # register the tool parser to ToolParserManager
    ToolParserManager.register_lazy_module(
        name="example",
        module_path="vllm.tool_parsers.example",
        class_name="ExampleToolParser",
    )

    ```

そのうえで、このプラグインをコマンドラインで次のように使えます。

```bash
    --enable-auto-tool-choice \
    --tool-parser-plugin <absolute path of the plugin file>
    --tool-call-parser example \
    --chat-template <your chat template> \
```
