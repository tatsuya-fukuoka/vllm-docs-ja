# リクエストごとのメトリクス { #per-request-metrics }

vLLM は、リクエストごとのタイミングに関するメトリクスを API レスポンスに直接含めて返せます。これは、`/metrics` で公開されるサーバー全体を集約した Prometheus メトリクスを補完するもので、課金や SLA モニタリング、個々のリクエスト単位のレイテンシ分析に役立ちます。

## 有効化 { #enabling }

`--enable-per-request-metrics` を付けてサーバーを起動します。

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-per-request-metrics
```

このフラグを指定すると、対応する API のレスポンスに、対象となる各リクエストのメトリクスが含まれます。

!!! note
    高い並行度のもとでは、リクエストごとのメトリクス計算を有効にすると無視できない
    CPU オーバーヘッドが生じる場合があります。本番で有効にする前に、実際のワークロードで
    ベンチマークを取って影響を評価してください。

## レスポンスの形式 { #response-format }

リクエストごとのメトリクスを有効にすると、レスポンスに `metrics` オブジェクトが含まれます。

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [ ... ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  },
  "metrics": {
    "time_to_first_token_ms": 85.2,
    "generation_time_ms": 1240.5,
    "queue_time_ms": 12.3,
    "mean_itl_ms": 9.1,
    "tokens_per_second": 103.2
  }
}
```

| フィールド | 説明 |
| --- | --- |
| `time_to_first_token_ms` | リクエストがスケジュールされてから最初の出力トークンが生成されるまでの時間（TTFT）。 |
| `generation_time_ms` | デコード時間。最初の出力トークンから最後の出力トークンまでの時間。キューでの待ち時間とプレフィル / TTFT はいずれも含みません。 |
| `queue_time_ms` | 処理が始まるまでにリクエストがスケジューラのキューで待った時間。 |
| `mean_itl_ms` | デコードフェーズにおける平均トークン間レイテンシ（連続する出力トークンの平均間隔）。出力が 1 トークンのみの場合は `null`。 |
| `tokens_per_second` | 出力トークンの全体スループット。推論区間（スケジュールから最後の出力トークンまで）における全生成トークン数を基準にします。`generation_time_ms` とは異なりプレフィルフェーズも含むため、純粋なデコード速度ではなくエンドツーエンドの生成速度を表します。 |

そのリクエストについて元となるタイミングデータが取得できない場合、すべてのフィールドは `null` になります。

!!! note
    タイミングのメトリクスは 1 本の生成ストリームを表すため、リクエストがちょうど 1 本に
    対応する場合にのみ返されます。`n > 1` のリクエストでは抑制されます（`metrics`
    オブジェクトが `null` になります）。これは、元のタイミングデータが `n` 本のシーケンスの
    うち 1 本しか反映しておらず、リクエスト全体に正確に紐づけられないためです。この場合でも
    トークン使用量（`prompt_tokens`、`completion_tokens`）は正確なままです。
    また、リクエストごとのメトリクスにはサーバー側の統計ログが必要で、これは既定で有効です。
    `--disable-log-stats` も同時に指定されている場合、vLLM は
    `--enable-per-request-metrics` を受け付けません。

## リクエストの例 { #example-request }

=== "Non-streaming"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    print(response.usage)
    print(response.model_extra.get("metrics"))
    ```

=== "Streaming"

    ストリーミングのレスポンスでは、メトリクスは最後の usage チャンク（すべてのコンテンツ
    チャンクの後に送られるチャンク）に付与されます。このチャンクは、
    `stream_options.include_usage: true` で usage の報告を有効にするか、サーバー側で
    `--enable-force-include-usage` によって強制した場合にのみ送出されます。強制していない
    場合、ストリーミングのクライアントはメトリクスを受け取るために
    `stream_options.include_usage: true` を設定する必要があります。

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.usage:
            print("Usage:", chunk.usage)
            print("Metrics:", chunk.model_extra.get("metrics"))
    ```

## Completions API { #completions-api }

リクエストごとのメトリクスは、同じ `metrics` レスポンスフィールドを使って `/v1/completions` エンドポイントでも利用できます。`n > 1` の場合と同様に、複数のプロンプトを含むリクエストではメトリクスは省略されます。タイミングデータを個々のプロンプトの生成に紐づけられないためです。

## Prometheus メトリクスとの関係 { #relationship-to-prometheus-metrics }

`metrics` レスポンスフィールドは、単一のリクエストについてリクエスト単位の値を提供します。一方、`/metrics` の Prometheus エンドポイントは、全リクエストにわたって集約したサーバーレベルのヒストグラム（`vllm:time_to_first_token_seconds` など）を公開します。
