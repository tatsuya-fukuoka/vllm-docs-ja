# 生成モデル { #generative-models }

vLLM は生成モデルを第一級でサポートしており、これはほとんどの LLM を含みます。

vLLM の生成モデルは [`VllmModelForTextGeneration`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/#vllm.model_executor.models.VllmModelForTextGeneration) インターフェイスを実装します。
これらのモデルは入力の最終的な隠れ状態にもとづいて、生成するトークンの対数確率を出力し、
それが [`Sampler`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/sample/sampler/#vllm.v1.sample.sampler.Sampler) を通って最終的なテキストになります。

## 設定 { #configuration }

### モデルランナー (`--runner`) { #model-runner-runner }

`--runner generate` オプションを指定すると、モデルを生成モードで実行します。

!!! tip
    vLLM は `--runner auto` で使用するモデルランナーを自動的に判定できるため、
    ほとんどの場合このオプションを設定する必要はありません。

## オフライン推論 { #offline-inference }

[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスは、オフライン推論のためのさまざまなメソッドを提供します。
モデル初期化時に指定できるオプションの一覧は[設定](https://docs.vllm.ai/en/v0.26.0/api/#configuration)（英語）を参照してください。

### `LLM.generate` { #llmgenerate }

[`generate`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.generate) メソッドは、vLLM のすべての生成モデルで利用できます。
[HF Transformers の同名メソッド](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate)に似ていますが、
トークナイズとデトークナイズも自動的に行われる点が異なります。

```python
from vllm import LLM

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate("Hello, my name is")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

[`SamplingParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.SamplingParams) を渡すことで、テキスト生成を制御することもできます。
たとえば `temperature=0` を設定すると貪欲サンプリングになります。

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
params = SamplingParams(temperature=0)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! important
    vLLM は既定で、Hugging Face のモデルリポジトリに `generation_config.json` があればそれを適用し、モデル作成者が推奨するサンプリングパラメータを使用します。多くの場合、[`SamplingParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.SamplingParams) を指定しなくてもこれが最良の結果をもたらします。

    vLLM 既定のサンプリングパラメータを使いたい場合は、[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) インスタンスの作成時に `generation_config="vllm"` を渡してください。
コード例はこちらにあります: [examples/basic/offline_inference/basic.py](../../examples/basic/offline_inference/basic.py)

### `LLM.beam_search` { #llmbeam_search }

[`beam_search`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.beam_search) メソッドは、[`generate`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.generate) の上に [ビームサーチ](https://huggingface.co/docs/transformers/en/generation_strategies#beam-search)を実装したものです。
たとえば、ビーム幅 5 で最大 50 トークンを出力する場合は次のようにします。

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")
params = BeamSearchParams(beam_width=5, max_tokens=50)
outputs = llm.beam_search([{"prompt": "Hello, my name is "}], params)

for output in outputs:
    generated_text = output.sequences[0].text
    print(f"Generated text: {generated_text!r}")
```

### `LLM.chat` { #llmchat }

[`chat`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.chat) メソッドは、[`generate`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM.generate) の上にチャット機能を実装したものです。
[OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) と同様の入力を受け取り、
モデルの[チャットテンプレート](https://huggingface.co/docs/transformers/en/chat_templating)を自動的に適用してプロンプトを組み立てます。

!!! important
    一般に、チャットテンプレートを持つのは instruction チューニング済みのモデルだけです。
    ベースモデルはチャット形式の会話に応答するよう学習されていないため、性能が低いことがあります。

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
        },
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

コード例はこちらにあります: [examples/basic/offline_inference/chat.py](../../examples/basic/offline_inference/chat.py)

モデルがチャットテンプレートを持たない場合や、別のテンプレートを指定したい場合は、
チャットテンプレートを明示的に渡せます。

```python
from vllm.entrypoints.chat_utils import load_chat_template

# You can find a list of existing chat templates under `examples/`
custom_template = load_chat_template(chat_template="<path_to_template>")
print("Loaded chat template:", custom_template)

outputs = llm.chat(conversation, chat_template=custom_template)
```

## オンラインサービング { #online-serving }

vLLM の [OpenAI 互換サーバー](../serving/online_serving/openai_compatible_server.md)は、オフライン API に対応するエンドポイントを提供します。

- [Completions API](../serving/online_serving/openai_compatible_server.md#completions-api) は `LLM.generate` に相当しますが、テキストのみを受け付けます。
- [Chat API](../serving/online_serving/openai_compatible_server.md#chat-api) は `LLM.chat` に相当し、チャットテンプレートを持つモデルであればテキストと[マルチモーダル入力](../features/multimodal_inputs.md)の両方を受け付けます。
