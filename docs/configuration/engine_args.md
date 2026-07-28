---
toc_depth: 3
---

# エンジン引数 { #engine-arguments }

エンジン引数は vLLM エンジンの挙動を制御します。

- [オフライン推論](../serving/offline_inference.md)では、[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスの引数の一部になります。
- [オンラインサービング](../serving/online_serving/README.md)では、`vllm serve` の引数の一部になります。

エンジン引数のクラスである [`EngineArgs`](https://docs.vllm.ai/en/v0.26.0/api/vllm/engine/arg_utils/#vllm.engine.arg_utils.EngineArgs) と [`AsyncEngineArgs`](https://docs.vllm.ai/en/v0.26.0/api/vllm/engine/arg_utils/#vllm.engine.arg_utils.AsyncEngineArgs) は、[vllm.config][] で定義された設定クラスを組み合わせたものです。開発者向けの情報を探している場合は、型・既定値・docstring の正となるこれらの設定クラスを参照することをおすすめします。

--8<-- "docs/cli/json_tip.inc.md"

## `EngineArgs` { #engineargs }

--8<-- "docs/generated/argparse/engine_args.inc.md"

## `AsyncEngineArgs` { #asyncengineargs }

--8<-- "docs/generated/argparse/async_engine_args.inc.md"
