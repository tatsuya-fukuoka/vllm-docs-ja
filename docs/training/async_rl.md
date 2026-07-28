# 非同期強化学習 { #async-reinforcement-learning }

## 概要 { #overview }

通常の RL の学習ループでは、生成と学習が順番に実行されます。ポリシーがロールアウトを生成し、そのロールアウトで学習を行い、これを繰り返します。生成中は学習用のアクセラレータが遊び、学習中は生成用のアクセラレータが遊びます。

**one-off パイプライン化**の手法では、生成と学習のフェーズを 2 つの並列なコルーチンに分離し、以前に生成したデータで学習しながら同時に新しいサンプルを生成できるようにします。これにより GPU の使用率とスループットが大きく向上します。

ただし、この重ね合わせには難しさがあります。リクエストの処理中に、推論エンジン側の重みを更新しなければならないためです。

## pause / resume API { #the-pause-and-resume-api }

推論エンジンの実行中に安全に重みを更新できるよう、vLLM は `pause_generation` と `resume_generation` のメソッドを提供しています。これらを使うと、実行中のリクエストを失うことなく、重み同期のための安全な時間枠をトレーナー側で確保できます。

### pause_generation { #pause_generation }

```python
await engine.pause_generation(mode="keep", clear_cache=True)
```

`mode` パラメータは、実行中のリクエストの扱いを制御します。

| モード | 挙動 |
| ---- | -------- |
| `"abort"` | 実行中のリクエストをすべて即座に中断し、部分的な結果を返す（既定） |
| `"wait"` | 実行中のリクエストがすべて完了するのを待ってから一時停止する |
| `"keep"` | リクエストをキューに凍結し、`resume_generation` の呼び出し時に再開する |

`clear_cache` パラメータは、一時停止後に KV キャッシュとプレフィックスキャッシュをクリアするかどうかを制御します。

### resume_generation { #resume_generation }

```python
await engine.resume_generation()
```

一時停止後にスケジューラを再開します。`mode="keep"` で凍結されたリクエストは生成を続けます。

### HTTP エンドポイント { #http-endpoints }

vLLM の HTTP サーバーを使う場合、同じ機能を次のエンドポイントから利用できます。

- `POST /pause?mode=keep` - 生成を一時停止する
- `POST /resume` - 生成を再開する
- `POST /abort_requests` - スケジューラを止めずに実行中のリクエストを中断する（すべて中断するには `{}`、個別には `{"request_ids": [...]}` を送る）

!!! note "データ並列の場合"
    vLLM の**内部ロードバランサー**（`data_parallel_backend="ray"`）を使ってデータ並列を行う場合、pause と resume はすべての DP ランクに対して自動的に処理されるため、1 回の呼び出しで十分です。**外部のロードバランサー**を使う場合は、各ランクに対して個別に呼び出す必要があります。

## 一般的な非同期 RL の流れ { #typical-async-rl-flow }

重み同期を伴う一般的な非同期 RL のループは次のようになります。

1. 現在のポリシーでロールアウトの生成を開始する
2. トレーナー側に新しい重みができたら、`mode="keep"` で生成を一時停止する
3. 更新後の重みをトレーナーから推論エンジンへ同期する（[重み転送](weight_transfer/README.md)を参照）
4. 生成を再開する。実行中のリクエストは新しい重みで続行される
5. これを繰り返す

重要な点は、`mode="keep"` で一時停止したリクエストは、停止前は**古い**重み、再開後は**新しい**重みからトークンを生成するということです。`clear_cache` パラメータは、KV キャッシュをクリアするかどうかを制御します。

## 例 { #example }

[非同期 RLHF の例](../../examples/rl/rlhf_async_new_apis.py)では、`vllm.AsyncLLMEngine`、NCCL による重み転送、実行中の pause / resume と検証を組み合わせたこのパターンを示しています。
