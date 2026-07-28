# 構造化出力 { #structured-outputs }

vLLM は、[xgrammar](https://github.com/mlc-ai/xgrammar) または
[guidance](https://github.com/guidance-ai/llguidance) をバックエンドとして、
構造化出力の生成をサポートしています。
このドキュメントでは、構造化出力を生成するためのさまざまなオプションの例を示します。

!!! warning
    v0.12.0 で削除された以下の非推奨 API フィールドをまだ使っている場合は、このドキュメントで示すように `structured_outputs` を使うようコードを更新してください。

    - `guided_json` -> `{"structured_outputs": {"json": ...}}` or `StructuredOutputsParams(json=...)`
    - `guided_regex` -> `{"structured_outputs": {"regex": ...}}` or `StructuredOutputsParams(regex=...)`
    - `guided_choice` -> `{"structured_outputs": {"choice": ...}}` or `StructuredOutputsParams(choice=...)`
    - `guided_grammar` -> `{"structured_outputs": {"grammar": ...}}` or `StructuredOutputsParams(grammar=...)`
    - `guided_whitespace_pattern` -> `{"structured_outputs": {"whitespace_pattern": ...}}` or `StructuredOutputsParams(whitespace_pattern=...)`
    - `structural_tag` -> `{"structured_outputs": {"structural_tag": ...}}` or `StructuredOutputsParams(structural_tag=...)`
    - `guided_decoding_backend` -> このフィールドをリクエストから削除してください

## オンラインサービング (OpenAI API) { #online-serving-openai-api }

OpenAI の [Completions](https://platform.openai.com/docs/api-reference/completions) API と [Chat](https://platform.openai.com/docs/api-reference/chat) API を使って構造化出力を生成できます。

次のパラメータがサポートされています。いずれも追加パラメータとして指定する必要があります。

- `choice`: 出力は選択肢のいずれか 1 つに厳密に一致します。
- `regex`: 出力は正規表現のパターンに従います。
- `json`: 出力は JSON スキーマに従います。
- `grammar`: 出力は文脈自由文法に従います。
- `structural_tag`: 生成テキスト内の指定したタグに囲まれた部分が JSON スキーマに従います。

サポートされているパラメータの一覧は [OpenAI 互換サーバー](../serving/online_serving/openai_compatible_server.md)のページを参照してください。

OpenAI 互換サーバーでは、構造化出力が既定でサポートされています。
使用するバックエンドは `vllm serve` の `--structured-outputs-config.backend` フラグで指定できます。
既定のバックエンドは `auto` で、リクエストの内容に応じて適切なバックエンドを選択しようとします。
特定のバックエンドとそのオプションを明示的に指定することもできます。
オプションの一覧は `vllm serve --help` で確認できます。

それでは各ケースの例を見ていきます。まずはもっとも簡単な `choice` からです。

??? code

    ```python
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="-",
    )
    model = client.models.list().data[0].id

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
        ],
        extra_body={"structured_outputs": {"choice": ["positive", "negative"]}},
    )
    print(completion.choices[0].message.content)
    ```

次の例は `regex` の使い方です。サポートされる正規表現の文法は構造化出力のバックエンドによって異なります。たとえば `xgrammar`、`guidance`、`outlines` は Rust 形式の正規表現を、`lm-format-enforcer` は Python の `re` モジュールを使います。ここでは、単純な正規表現テンプレートからメールアドレスを生成します。

??? code

    ```python
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an example email address for Alan Turing, who works in Enigma. End in .com and new line. Example result: alan.turing@enigma.com\n",
            }
        ],
        extra_body={"structured_outputs": {"regex": r"\w+@\w+\.com\n"}, "stop": ["\n"]},
    )
    print(completion.choices[0].message.content)
    ```

構造化テキスト生成でもっとも重要な機能の 1 つが、あらかじめ定義したフィールドと形式に沿った正しい JSON を生成できることです。
これには `json` パラメータを 2 通りの方法で使えます。

- [JSON Schema](https://json-schema.org/) を直接指定する
- [Pydantic モデル](https://docs.pydantic.dev/latest/)を定義し、そこから JSON Schema を取り出す（通常はこちらのほうが簡単です）

次の例は、Pydantic モデルと `response_format` パラメータを組み合わせる方法です。

??? code

    ```python
    from pydantic import BaseModel
    from enum import Enum

    class CarType(str, Enum):
        sedan = "sedan"
        suv = "SUV"
        truck = "Truck"
        coupe = "Coupe"

    class CarDescription(BaseModel):
        brand: str
        model: str
        car_type: CarType

    json_schema = CarDescription.model_json_schema()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "car-description",
                "schema": CarDescription.model_json_schema()
            },
        },
    )
    print(completion.choices[0].message.content)
    ```

!!! tip
    必須ではありませんが、JSON スキーマと各フィールドの埋め方をプロンプトにも書いておくとよいでしょう。
    多くの場合、これで結果が大きく改善します。

最後は `grammar` オプションです。使いこなすのはもっとも難しいですが、非常に強力です。
SQL クエリのような言語全体を定義でき、文脈自由な EBNF 文法によって動作します。
例として、簡略化した SQL クエリの形式を定義してみます。

??? code

    ```python
    simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate an SQL query to show the 'username' and 'email' from the 'users' table.",
            }
        ],
        extra_body={"structured_outputs": {"grammar": simplified_sql_grammar}},
    )
    print(completion.choices[0].message.content)
    ```

参考: [完全な例](https://docs.vllm.ai/en/v0.26.0/examples/features/structured_outputs/)（英語）

## Reasoning 出力 { #reasoning-outputs }

reasoning モデルでは、<project:#reasoning-outputs> と構造化出力を組み合わせることもできます。

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
```

reasoning はどの構造化出力の機能とも組み合わせられます。次の例では JSON スキーマと併用しています。

??? code

    ```python
    from pydantic import BaseModel


    class People(BaseModel):
        name: str
        age: int


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Generate a JSON with the name and age of one random person.",
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "people",
                "schema": People.model_json_schema()
            }
        },
    )
    print("reasoning: ", completion.choices[0].message.reasoning)
    print("content: ", completion.choices[0].message.content)
    ```

参考: [完全な例](https://docs.vllm.ai/en/v0.26.0/examples/features/structured_outputs/)（英語）

!!! note
    Qwen3 Coder 系のモデルで reasoning を有効にした場合、reasoning の内容が `reasoning` フィールドに分離して解析されないと、構造化出力が無効になることがあります（v0.11.2 以降）。
    両方の機能を同時に使うには、reasoning モードでの構造化出力を明示的に有効にする必要があります。
    vLLM サーバーの起動時に `--structured-outputs-config.enable_in_reasoning=True` を追加してください。
    参考: [Reasoning 出力](reasoning_outputs.md)のドキュメント。

## 実験的な自動パース (OpenAI API) { #experimental-automatic-parsing-openai-api }

このセクションでは、`client.chat.completions.create()` を包む OpenAI のベータ版ラッパーについて説明します。Python の型とより密に連携できます。

執筆時点（`openai==1.54.4`）では、これは OpenAI クライアントライブラリの「ベータ」機能です。コードは[こちら](https://github.com/openai/openai-python/blob/52357cff50bee57ef442e94d78a0de38b4173fc2/src/openai/resources/beta/chat/completions.py#L100-L104)を参照してください。

以下の例では、`vllm serve meta-llama/Llama-3.1-8B-Instruct` で vLLM を起動しています。

Pydantic モデルを使って構造化出力を得る簡単な例です。

??? code

    ```python
    from pydantic import BaseModel
    from openai import OpenAI

    class Info(BaseModel):
        name: str
        age: int

    client = OpenAI(base_url="http://0.0.0.0:8000/v1", api_key="dummy")
    model = client.models.list().data[0].id
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Cameron, I'm 28. What's my name and age?"},
        ],
        response_format=Info,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    print("Name:", message.parsed.name)
    print("Age:", message.parsed.age)
    ```

```console
ParsedChatCompletionMessage[Testing](content='{"name": "Cameron", "age": 28}', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=Testing(name='Cameron', age=28))
Name: Cameron
Age: 28
```

ネストした Pydantic モデルを使い、数学の解法を段階的に扱うより複雑な例です。

??? code

    ```python
    from typing import List
    from pydantic import BaseModel
    from openai import OpenAI

    class Step(BaseModel):
        explanation: str
        output: str

    class MathResponse(BaseModel):
        steps: list[Step]
        final_answer: str

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful expert math tutor."},
            {"role": "user", "content": "Solve 8x + 31 = 2."},
        ],
        response_format=MathResponse,
    )

    message = completion.choices[0].message
    print(message)
    assert message.parsed
    for i, step in enumerate(message.parsed.steps):
        print(f"Step #{i}:", step)
    print("Answer:", message.parsed.final_answer)
    ```

出力:

```console
ParsedChatCompletionMessage[MathResponse](content='{ "steps": [{ "explanation": "First, let\'s isolate the term with the variable \'x\'. To do this, we\'ll subtract 31 from both sides of the equation.", "output": "8x + 31 - 31 = 2 - 31"}, { "explanation": "By subtracting 31 from both sides, we simplify the equation to 8x = -29.", "output": "8x = -29"}, { "explanation": "Next, let\'s isolate \'x\' by dividing both sides of the equation by 8.", "output": "8x / 8 = -29 / 8"}], "final_answer": "x = -29/8" }', refusal=None, role='assistant', audio=None, function_call=None, tool_calls=[], parsed=MathResponse(steps=[Step(explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation.", output='8x + 31 - 31 = 2 - 31'), Step(explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.', output='8x = -29'), Step(explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8.", output='8x / 8 = -29 / 8')], final_answer='x = -29/8'))
Step #0: explanation="First, let's isolate the term with the variable 'x'. To do this, we'll subtract 31 from both sides of the equation." output='8x + 31 - 31 = 2 - 31'
Step #1: explanation='By subtracting 31 from both sides, we simplify the equation to 8x = -29.' output='8x = -29'
Step #2: explanation="Next, let's isolate 'x' by dividing both sides of the equation by 8." output='8x / 8 = -29 / 8'
Answer: x = -29/8
```

`structural_tag` の使用例はこちらにあります: [examples/features/structured_outputs](https://docs.vllm.ai/en/v0.26.0/examples/features/structured_outputs/)（英語）

## オフライン推論 { #offline-inference }

オフライン推論でも同じ種類の構造化出力を利用できます。
利用するには、`SamplingParams` の中で `StructuredOutputsParams` クラスを使って構造化出力を設定します。
`StructuredOutputsParams` で指定できる主なオプションは次のとおりです。

- `json`
- `regex`
- `choice`
- `grammar`
- `structural_tag`

これらのパラメータは、上記のオンラインサービングの例と同じように使えます。
`choice` パラメータの使用例を次に示します。

??? code

    ```python
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    structured_outputs_params = StructuredOutputsParams(choice=["Positive", "Negative"])
    sampling_params = SamplingParams(structured_outputs=structured_outputs_params)
    outputs = llm.generate(
        prompts="Classify this sentiment: vLLM is wonderful!",
        sampling_params=sampling_params,
    )
    print(outputs[0].outputs[0].text)
    ```

参考: [完全な例](../../examples/features/structured_outputs/structured_outputs_offline.py)
