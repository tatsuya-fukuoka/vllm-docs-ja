# ユニットテスト { #unit-testing }

このページでは、実装したモデルを検証するためのユニットテストの書き方を説明します。

## 必須のテスト { #required-tests }

これらのテストは、PR を vLLM ライブラリにマージしてもらうために必要です。これらがないと、PR の CI が失敗します。

### モデルの読み込み { #model-loading }

[tests/models/registry.py](../../../tests/models/registry.py) に、そのモデルの HuggingFace リポジトリの例を追加してください。これにより、ダミーの重みを読み込んで vLLM でモデルを初期化できることを確認するユニットテストが有効になります。

!!! important
    各セクションのモデル一覧はアルファベット順に保ってください。

!!! tip
    モデルが HF Transformers の開発版を必要とする場合は、`min_transformers_version` を設定すると、そのバージョンがリリースされるまで CI 上でテストをスキップできます。

## 任意のテスト { #optional-tests }

これらのテストは、PR を vLLM ライブラリにマージしてもらうために必須ではありません。ただし、これらのテストが通ると実装が正しいという確信が高まり、将来のリグレッションを防ぐのに役立ちます。

### モデルの正しさ { #model-correctness }

これらのテストでは、vLLM のモデル出力を [HF Transformers](https://github.com/huggingface/transformers) と比較します。新しいテストは [tests/models](../../../tests/models) のサブディレクトリに追加できます。

#### 生成モデル { #generative-models }

[生成モデル](../../models/generative_models.md)については、[tests/models/utils.py](../../../tests/models/utils.py) で定義されているとおり、正しさのテストに 2 つのレベルがあります。

- 完全一致（`check_outputs_equal`）: vLLM が出力するテキストが HF の出力と完全に一致すること。
- logprobs の類似度（`check_logprobs_close`）: vLLM が出力する logprobs が HF の出力する top-k の logprobs に含まれ、その逆も成り立つこと。

#### プーリングモデル { #pooling-models }

[プーリングモデル](../../models/pooling_models/README.md)については、[tests/models/utils.py](../../../tests/models/utils.py) で定義されているとおり、単純にコサイン類似度を確認します。

### マルチモーダル処理 { #multi-modal-processing }

#### 共通テスト { #common-tests }

[tests/models/multimodal/processing/test_common.py](../../../tests/models/multimodal/processing/test_common.py) にモデルを追加すると、次の入力の組み合わせがすべて同じ出力になることを検証できます。

- テキスト + マルチモーダルデータ
- トークン + マルチモーダルデータ
- テキスト + キャッシュ済みマルチモーダルデータ
- トークン + キャッシュ済みマルチモーダルデータ

#### モデル固有のテスト { #model-specific-tests }

そのモデルにのみ当てはまるテストを実行するには、[tests/models/multimodal/processing](../../../tests/models/multimodal/processing) の下に新しいファイルを追加できます。

たとえば、モデルの HF プロセッサがユーザー指定のキーワード引数を受け取る場合、[tests/models/multimodal/processing/test_phi3v.py](../../../tests/models/multimodal/processing/test_phi3v.py) のように、そのキーワード引数が正しく適用されているかを検証できます。
