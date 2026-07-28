# カスタム引数 { #custom-arguments }

vLLM の*カスタム引数*を使うと、vLLM の `SamplingParams` や REST API の仕様に含まれない引数を渡せます。カスタム引数は辞書として渡されるため、追加や削除に vLLM の再コンパイルは不要です。

たとえば、vLLM のソースコードを変更せずに[独自の logits プロセッサ](./custom_logitsprocs.md)を使いたい場合に便利です。

!!! note
    独自の logits プロセッサでは、カスタム引数に対する `validate_params` を必ず実装してください。実装しないと、不正なカスタム引数が予期しない動作を引き起こす可能性があります。

## オフラインでのカスタム引数 { #offline-custom-arguments }

`SamplingParams.extra_args` に `dict` として渡したカスタム引数は、`SamplingParams` にアクセスできるあらゆるコードから参照できます。

``` python
SamplingParams(extra_args={"your_custom_arg_name": 67})
```

これにより、`SamplingParams` に含まれていない引数をリクエストの一部として `LLM` に渡せます。

## オンラインでのカスタム引数 { #online-custom-arguments }

vLLM の REST API では、`vllm_xargs` を通じてカスタム引数を vLLM サーバーに渡せます。次の例は、REST API のリクエストにカスタム引数を組み込んだものです。

``` bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        ...
        "vllm_xargs": {"your_custom_arg": 67}
    }'
```

OpenAI の SDK を使う場合は、`extra_body` 引数から `vllm_xargs` を指定できます。

``` python
batch = await client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    ...,
    extra_body={
        "vllm_xargs": {
            "your_custom_arg": 67
        }
    }
)
```

!!! note
    `vllm_xargs` は内部的に `SamplingParams.extra_args` に代入されるため、`SamplingParams.extra_args` を使うコードはオフライン・オンラインの両方でそのまま動作します。
