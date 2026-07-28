# 本番運用のメトリクス { #production-metrics }

vLLM は、システムの健全性を監視するための各種メトリクスを公開しています。
これらは vLLM の OpenAI 互換 API サーバーの `/metrics` エンドポイントから取得できます。

サーバーは Python から、あるいは [Docker](../deployment/docker.md) を使って起動できます。

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

起動したら、エンドポイントに問い合わせて最新のメトリクスを取得します。

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

公開されているメトリクスは次のとおりです。

## 一般的なメトリクス { #general-metrics }

--8<-- "docs/generated/metrics/general.inc.md"

## 投機的デコーディングのメトリクス { #speculative-decoding-metrics }

--8<-- "docs/generated/metrics/spec_decode.inc.md"

## NIXL KV コネクタのメトリクス { #nixl-kv-connector-metrics }

--8<-- "docs/generated/metrics/nixl_connector.inc.md"

## Model Flops Utilization (MFU) の性能メトリクス { #model-flops-utilization-mfu-performance-metrics }

これらのメトリクスは `--enable-mfu-metrics` を指定すると利用できます。

--8<-- "docs/generated/metrics/perf.inc.md"

## 非推奨化の方針 { #deprecation-policy }

補足: バージョン `X.Y` で非推奨になったメトリクスは、`X.Y+1` では非表示になりますが
`--show-hidden-metrics-for-version=X.Y` を指定すれば再度有効にできます。その後 `X.Y+2` で削除されます。
