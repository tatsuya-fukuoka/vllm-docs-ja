# オフライン推論 { #offline-inference }

vLLM の [`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスを使うと、自分のコードの中でオフライン推論を実行できます。

## モデルの種類 { #model-types }

vLLM のモデルは 2 種類に分類できます。

- **[生成モデル](../models/supported_models.md)** - テキストの補完やチャットの応答を生成するモデル（LLaMA、Qwen、DeepSeek など）。これらのモデルでは `LLM.generate()` と `LLM.chat()` を使います。

- **[プーリングモデル](../models/pooling_models/README.md)** - コンテンツを生成しないモデル。bge-m3 や Qwen3 Reranker のように、主に分類や検索のタスクに使われます。

## 生成系 API { #generative-apis }

生成モデルの詳細は[このページ](../models/supported_models.md)を参照してください。

- `LLM.generate` - 与えられた入力プロンプトに対する補完を生成します。
- `LLM.chat` - チャットの会話に対する応答を生成します。

## 非同期キュー API { #asynchronous-queue-apis }

- `LLM.enqueue` - 完了を待たずに、生成対象のプロンプトをキューに追加します。
- `LLM.enqueue_chat` - 完了を待たずに、生成対象のチャット会話をキューに追加します。
- `LLM.wait_for_completion` - キューに入れたすべてのリクエストの完了を待ち、結果を返します。

## プーリング API { #pooling-apis }

プーリングモデルの詳細は[このページ](../models/pooling_models/README.md)を参照してください。

- `LLM.classify` - [分類モデル](../models/pooling_models/classify.md)にのみ適用できます。
- `LLM.embed` - [埋め込みモデル](../models/pooling_models/embed.md)にのみ適用できます。
- `LLM.score` - [スコアモデル](../models/pooling_models/scoring.md)（cross-encoder、bi-encoder、late-interaction）に適用できます。
- `LLM.encode` - すべての[プーリングモデル](../models/pooling_models/README.md)に適用できます。

## プロファイリング API { #profiling-apis }

プロファイリングの詳細は[このページ](../contributing/profiling.md)を参照してください。

- `LLM.start_profile` - 任意のトレース接頭辞を指定してプロファイリングを開始します。
- `LLM.stop_profile` - 実行中のプロファイリングセッションを停止します。

## スリープモード API { #sleep-mode-apis }

スリープモードの詳細は[このページ](../features/sleep_mode.md)を参照してください。

- `LLM.sleep` - エンジンをスリープモードにします。
- `LLM.wake_up` - スリープモードからエンジンを復帰させます。

## キャッシュ管理 API { #cache-management-apis }

- `LLM.reset_mm_cache` - マルチモーダルキャッシュをリセットします。
- `LLM.reset_prefix_cache` - プレフィックスキャッシュをリセットします。

## メトリクス API { #metrics-apis }

メトリクスの詳細は[このページ](../design/metrics.md)を参照してください。

- `LLM.get_metrics` - Prometheus 形式で集計されたメトリクスのスナップショットを返します。

## 重み転送 API（RL 学習） { #weight-transfer-apis-rl-training }

重み転送の詳細は[このページ](../training/weight_transfer/README.md)を参照してください。

- `LLM.init_weight_transfer_engine` - RL 学習用の重み転送エンジンを初期化します。
- `LLM.start_weight_update` - 新しい重み更新サイクルを開始します。
- `LLM.update_weights` - モデルの重みを更新します。
- `LLM.finish_weight_update` - 現在の重み更新サイクルを終了します。

## その他の API { #additional-apis }

- `LLM.collective_rpc` - すべてのワーカーに対してメソッドや呼び出し可能オブジェクトを一斉に実行します。
- `LLM.apply_model` - 各ワーカー内のモデルに直接関数を適用します。

## API リファレンス { #api-reference }

[Offline Inference](https://docs.vllm.ai/en/v0.26.0/api/#offline-inference)（英語）

## Ray Data LLM API { #ray-data-llm-api }

Ray Data LLM は、vLLM を内部エンジンとして利用する、もう 1 つのオフライン推論 API です。
この API には、大規模かつ GPU 効率の高い推論を簡単にするための機能が一通り備わっています。

- ストリーミング実行により、クラスタ全体のメモリを超えるデータセットも処理できます。
- 自動シャーディング・負荷分散・オートスケーリングにより、耐障害性を備えた形で Ray クラスタ全体に処理を分散します。
- 連続バッチングにより vLLM のレプリカを飽和状態に保ち、GPU 使用率を最大化します。
- テンソル並列・パイプライン並列を透過的にサポートし、効率的なマルチ GPU 推論を実現します。
- 主要なファイル形式とクラウドオブジェクトストレージの読み書きに対応しています。
- コードを変更せずにワークロードをスケールアップできます。

??? code

    ```python
    import ray  # Requires ray>=2.44.1
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

    config = vLLMEngineProcessorConfig(model_source="unsloth/Llama-3.2-1B-Instruct")
    processor = build_llm_processor(
        config,
        preprocess=lambda row: {
            "messages": [
                {"role": "system", "content": "You are a bot that completes unfinished haikus."},
                {"role": "user", "content": row["item"]},
            ],
            "sampling_params": {"temperature": 0.3, "max_tokens": 250},
        },
        postprocess=lambda row: {"answer": row["generated_text"]},
    )

    ds = ray.data.from_items(["An old silent pond..."])
    ds = processor(ds)
    ds.write_parquet("local:///tmp/data/")
    ```

Ray Data LLM API の詳細は [Ray Data LLM のドキュメント](https://docs.ray.io/en/latest/data/working-with-llms.html)（英語）を参照してください。
