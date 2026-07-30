# CoreWeave の Tensorizer によるモデルの読み込み { #loading-models-with-coreweaves-tensorizer }

vLLM は [CoreWeave の Tensorizer](https://docs.coreweave.com/coreweave-machine-learning-and-ai/inference/tensorizer) を使ったモデルの読み込みをサポートしています。ディスク、HTTP/HTTPS エンドポイント、あるいは S3 エンドポイントにシリアライズされた vLLM のモデルテンソルを、実行時に GPU へ直接、きわめて高速にデシリアライズできます。その結果、Pod の起動時間と CPU メモリの使用量を大幅に削減できます。テンソルの暗号化にも対応しています。

vLLM は Tensorizer をモデル読み込みの仕組みに完全に統合しています。以下では、vLLM で Tensorizer を使い始める方法を簡単に説明します。

## Tensorizer のインストール { #installing-tensorizer }

`tensorizer` をインストールするには `pip install vllm[tensorizer]` を実行します。

## 基本 { #the-basics }

Tensorizer を使ってモデルを読み込むには、まずそのモデルを Tensorizer でシリアライズする必要があります。この処理は[サンプルスクリプト](../../../examples/features/tensorize_vllm_model.py)が担当します。

ここでは、このスクリプトで `facebook/opt-125m` をシリアライズし、それを推論用に読み込むという基本的な流れを見ていきます。

## Tensorizer で vLLM モデルをシリアライズする { #serializing-a-vllm-model-with-tensorizer }

Tensorizer でモデルをシリアライズするには、必要な CLI 引数を付けてサンプルスクリプトを呼び出します。スクリプト自身の docstring に CLI 引数と正しい使い方が詳しく説明されています。ここでは、モデルを S3 バケット `s3://my-bucket` にシリアライズして保存したいものとして、docstring の例をそのまま使います。

```bash
python examples/features/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

これでモデルのテンソルが `s3://my-bucket/vllm/facebook/opt-125m/v1` に保存されます。tensorize したモデルに LoRA アダプターを適用する予定がある場合は、上のコマンドで LoRA アダプターの HF ID を渡すと、その成果物も同じ場所に保存されます。

```bash
python examples/features/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --suffix v1
```

## Tensorizer を使ったモデルのサービング { #serving-the-model-using-tensorizer }

モデルを目的の場所にシリアライズしたら、`vllm serve` または `LLM` エントリポイントで読み込めます。保存先のディレクトリを `LLM()` および `vllm serve` の `model` 引数に渡してください。たとえば、LoRA アダプター付きで先ほど保存した tensorize 済みモデルをサービングするには次のようにします。

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora 
```

`LLM()` を使う場合は次のとおりです。

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
)
```

## Tensorizer の設定オプション { #options-for-configuring-tensorizer }

`tensorizer` でモデルのシリアライズとデシリアライズを担う中心的なオブジェクトは、それぞれ `TensorSerializer` と `TensorDeserializer` です。これらに任意のキーワード引数を渡してシリアライズ / デシリアライズの挙動を設定するには、`model_loader_extra_config` のキーとして、それぞれ `serialization_kwargs` と `deserialization_kwargs` を指定します。上記オブジェクトの全パラメータを説明した完全な docstring は、`tensorizer` の [serialization.py](https://github.com/coreweave/tensorizer/blob/main/tensorizer/serialization.py) にあります。

たとえば、`tensorizer` でシリアライズする際の CPU の並行度は、`TensorSerializer` のイニシャライザの `limit_cpu_concurrency` パラメータで制限できます。`limit_cpu_concurrency` を任意の値に設定するには、シリアライズ時に次のようにします。

```bash
python examples/features/tensorize_vllm_model.py \
   --model facebook/opt-125m \
   --lora-path <lora_id> \
   serialize \
   --serialized-directory s3://my-bucket \
   --serialization-kwargs '{"limit_cpu_concurrency": 2}' \
   --suffix v1
```

`TensorDeserializer` を通じて読み込み処理をカスタマイズする例として、`model_loader_extra_config` 経由でイニシャライザの `num_readers` パラメータを指定し、デシリアライズ時の同時リーダー数を制限できます。

```bash
vllm serve s3://my-bucket/vllm/facebook/opt-125m/v1 \
    --load-format tensorizer \
    --enable-lora \
    --model-loader-extra-config '{"deserialization_kwargs": {"num_readers": 2}}'
```

`LLM()` を使う場合は次のとおりです。

```python
from vllm import LLM
llm = LLM(
    "s3://my-bucket/vllm/facebook/opt-125m/v1", 
    load_format="tensorizer",
    enable_lora=True,
    model_loader_extra_config={"deserialization_kwargs": {"num_readers": 2}},
)
```
