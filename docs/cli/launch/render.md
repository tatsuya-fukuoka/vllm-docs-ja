# vllm launch render { #vllm-launch-render }

## 概要 { #overview }

`vllm launch render` は、前処理と後処理だけを行う GPU 不要のレンダリングサーバーを起動します。

```bash
vllm launch render meta-llama/Llama-3.2-1B-Instruct --port 8100
```

このコマンドは通常のサービング用のパーサーを再利用しているため、モデル・フロントエンド・
ネットワークなどの CLI オプションは [`vllm serve`](../serve.md) と同じ規則に従います。

## JSON 形式の CLI 引数 { #json-cli-arguments }

--8<-- "docs/cli/json_tip.inc.md"

## 引数 { #arguments }

--8<-- "docs/generated/argparse/launch_render.inc.md"
