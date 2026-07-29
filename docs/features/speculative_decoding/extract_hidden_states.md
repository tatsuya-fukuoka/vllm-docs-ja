# 隠れ状態の抽出 { #hidden-state-extraction }

隠れ状態の抽出（Hidden State Extraction）機能を使うと、推論中にターゲットモデルの中間層の活性値を vLLM に保存させられます。[EAGLE](eagle.md) 形式のドラフトモデルの学習、知識蒸留、モデル内部のオフライン分析などに役立ちます。

!!! note
    レイヤー ID として `num_hidden_layers` を渡すと、最終層の出力隠れ状態を保存できます。これらは
    出力側の正規化（output norm）が適用されて_いない_点に注意してください。

## オフラインの例 { #offline-example }

```python
import tempfile

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)

with tempfile.TemporaryDirectory() as tmpdir:
    llm = LLM(
        model="Qwen/Qwen3-8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4],
                },
            },
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": tmpdir,
            },
        ),
    )

    outputs = llm.generate(
        ["The future of AI is"],
        SamplingParams(max_tokens=1),
    )

    for output in outputs:
        path = output.kv_transfer_params["hidden_states_path"]
        obj = example_hidden_states_connector.load_hidden_states(path)
        print(f"token_ids: {obj['token_ids'].shape}")
        print(f"hidden_states: {obj['hidden_states'].shape}")
```

完全な例は [`examples/features/speculative_decoding/extract_hidden_states_offline.py`](../../../examples/features/speculative_decoding/extract_hidden_states_offline.py) にあります。

## オンラインの例 { #online-example }

性能を上げるため、オンラインで利用する場合は `/dev/shm/` のような RAM 上のファイルシステムを使い、クライアントが生成後すぐにファイルを削除する運用を推奨します。

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{"method": "extract_hidden_states", "num_speculative_tokens": 1, "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3, 4]}}}' \
    --kv_transfer_config '{"kv_connector": "ExampleHiddenStatesConnector", "kv_role": "kv_producer", "kv_connector_extra_config": {"shared_storage_path": "/dev/shm/hidden_states"}}'
```

## リクエストごとのオプション { #per-request-options }

オフライン・オンラインのどちらのモードでも、`kv_transfer_params` を通じてリクエストごとのオプションを指定できます。

| パラメータ | 既定値 | 説明 |
| --- | --- | --- |
| `hidden_states_path` | 自動生成 | 隠れ状態を保存するファイルパスを指定します。未設定の場合は `<shared_storage_path>/<request_id>.safetensors` に保存されます。サーバー設定で `allow_custom_save_path` が有効になっている必要があります。 |
| `include_output_tokens` | `False` | `True` の場合、プロンプトと生成された出力トークンの両方について隠れ状態を保存します。`False` の場合はプロンプトのトークンの隠れ状態のみを保存します。 |

### オフラインでの使い方 { #offline-usage }

リクエストごとのオプションは `SamplingParams` の `extra_args` で渡します。

```python
SamplingParams(
    max_tokens=32,
    extra_args={
        "kv_transfer_params": {
            "hidden_states_path": "/tmp/my_output.safetensors",
            "include_output_tokens": True,
        }
    },
)
```

### オンラインでの使い方 { #online-usage }

API リクエストのトップレベルのフィールドとして `kv_transfer_params` を渡します。

```json
{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "kv_transfer_params": {
        "hidden_states_path": "/tmp/my_output.safetensors",
        "include_output_tokens": true
    }
}
```

## 設定 { #configuration }

`kv_connector_extra_config` の辞書では、次のサーバーレベルのオプションを指定できます。

| パラメータ | 既定値 | 説明 |
| --- | --- | --- |
| `shared_storage_path` | `/tmp` | 隠れ状態のファイルを保存するディレクトリ（リクエストごとに `hidden_states_path` が指定されていない場合に使用） |
| `allow_custom_save_path` | `False` | API クライアントが `hidden_states_path` で任意のファイルパスを指定できるようにします。無効の場合、クライアントが指定したパスは警告とともに無視されます。任意のパスはサーバー上の任意の場所に書き込めるため、信頼できるクライアントに対してのみ有効にしてください。 |
| `num_writer_threads` | `8` | 非同期のディスク書き込みに使うスレッドプールのサイズ |
| `use_synchronization_lock` | `True` | ファイルロックを使い、書き込みが完了するまで同時に読み込むプロセスをブロックします。同期が不要なバッチ生成では無効にできます。 |

## 出力形式 { #output-format }

各リクエストは、次を含む `.safetensors` ファイルを生成します。

- **`hidden_states`** — 形状 `[num_tokens, num_extracted_layers, hidden_size]`
- **`token_ids`** — 形状 `[num_tokens]`

ファイルパスは `output.kv_transfer_params["hidden_states_path"]` で返されます。適切な同期のもとでファイルを読むには、コネクタモジュールの `load_hidden_states()` を使ってください。

!!! note
    チャンク化プレフィルはこの機能と併用できないため、無効にする必要があります。
