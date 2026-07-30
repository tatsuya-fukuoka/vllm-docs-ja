# モデルの登録 { #registering-a-model }

vLLM は、各モデルの実行方法を判断するためにモデルレジストリを利用します。
登録済みのアーキテクチャの一覧は[こちら](../../models/supported_models.md)にあります。

対象のモデルがこの一覧にない場合は、vLLM に登録する必要があります。
このページでは、その具体的な手順を説明します。

## 組み込みモデル { #built-in-models }

モデルを vLLM のライブラリに直接追加するには、まず [GitHub リポジトリ](https://github.com/vllm-project/vllm)をフォークし、[ソースからビルド](../../getting_started/installation/gpu.md#build-wheel-from-source)します。
これでコードベースを変更し、モデルをテストできるようになります。

モデルを実装したら（[チュートリアル](basic.md)を参照）、[vllm/model_executor/models](../../../vllm/model_executor/models) ディレクトリに配置します。
次に、[vllm/model_executor/models/registry.py](../../../vllm/model_executor/models/registry.py) の `_VLLM_MODELS` にモデルクラスを追加すると、vLLM の import 時に自動的に登録されます。
最後に、[対応モデルの一覧](../../models/supported_models.md)を更新して、あなたのモデルを紹介しましょう。

!!! important
    各セクションのモデルの一覧はアルファベット順に保ってください。

## ツリー外のモデル { #out-of-tree-models }

vLLM のコードベースを変更せずに、[プラグイン](../../design/plugin_system.md)を使って外部のモデルを読み込めます。

モデルを登録するには、次のコードを使います。

```python
# The entrypoint of your plugin
def register():
    from vllm import ModelRegistry
    from your_code import YourModelForCausalLM

    ModelRegistry.register_model("YourModelForCausalLM", YourModelForCausalLM)
```

モデルが CUDA を初期化するモジュールを import している場合は、`RuntimeError: Cannot re-initialize CUDA in forked subprocess` のようなエラーを避けるため、遅延 import を検討してください。

```python
# The entrypoint of your plugin
def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "YourModelForCausalLM",
        "your_code:YourModelForCausalLM",
    )
```

!!! important
    マルチモーダルのモデルの場合は、モデルクラスが [`SupportsMultiModal`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal) インターフェイスを実装していることを確認してください。
    詳細は[こちら](multimodal.md)を参照してください。
