# IPC エンジン { #ipc-engine }

IPC の重み転送エンジンは、**CUDA IPC**（プロセス間通信）のハンドルを使って、**同じ GPU** 上のトレーナーと推論ワーカーの間で GPU メモリを直接共有します。データのコピーが一切発生しないため、学習と推論を同居させる場合にもっとも効率的な選択肢です。複数 GPU の構成もサポートされており、各 GPU が重みを all-gather し、同居する適切なプロセスが取り出します。

## IPC を使う場面 { #when-to-use-ipc }

- 学習と推論が**同じ GPU** を共有する（同居構成）場合

## 仕組み { #how-it-works }

1. トレーナーが各重みの CUDA テンソルを作り、`torch.multiprocessing.reductions.reduce_tensor` で IPC ハンドルを生成します。複数 GPU の構成（FSDP など）では、各トレーナーランクが IPC ハンドルを生成する前に、各層の完全なテンソルを自分の GPU 上に all-gather する必要があります。
2. 各 GPU の IPC ハンドルは、**Ray**・**HTTP**・**独自の呼び出し可能オブジェクト**のいずれかで推論エンジンへ送られます。各ランクは自分の GPU に対応するハンドルだけを読み取ります。
3. 推論ワーカーは `rebuild_cuda_tensor` でハンドルからテンソルを再構成し、トレーナーの GPU メモリを直接参照します。

!!! warning
    IPC ハンドルのやり取りはシリアライズされた Python オブジェクトの送信を伴います。HTTP で転送する場合は、サーバーとクライアントの両方で `VLLM_ALLOW_INSECURE_SERIALIZATION=1` を設定する必要があります。HTTP 送信のために IPC ハンドルが pickle 化・base64 エンコードされるためです。

## パック（チャンク）転送 { #packed-chunked-transfer }

既定では、すべての重みを 1 回の API 呼び出しで送ります。大きなモデルでは、両側の GPU メモリにモデル全体が同時に載っている必要があります。`packed=True` を設定すると、GPU メモリの使用量を抑えた**チャンク転送**が有効になります。

- 重みは固定サイズのパックバッファに連結されます（`packed_buffer_size_bytes` で制御）。
- 各チャンクは、1 組の `start_weight_update` / `finish_weight_update` の中で個別の `update_weights` 呼び出しとして送られます。そのため層単位の再読み込みは、チャンク数によらず最初に 1 回初期化され、最後に 1 回確定されます。
- 各チャンクを消費した後、そのチャンクの GPU メモリを解放できます。

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="ray",
    llm_handle=llm_actor_handle,
    packed=True,
    packed_buffer_size_bytes=256 * 1024 * 1024,  # 256 MB chunks
)
```

## 初期化 { #initialization }

IPC バックエンドでは、どちら側にも初期化は不要です。`init_transfer_engine` の呼び出しは IPC では何もしません。

## 重みの送信 { #sending-weights }

IPC では、ハンドルを届けるために 2 つの転送モードをサポートしています。

### Ray モード { #ray-mode }

vLLM を Ray のアクターとして動かす場合に使います。

```python
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="ray",
    llm_handle=llm_actor_handle,
)
# start
ray.get(llm_actor_handle.start_weight_update.remote())
# send weights
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
# finish
ray.get(llm_actor_handle.finish_weight_update.remote())
```

Ray モードでは、エンジンが `llm_handle.update_weights.remote(...)` を直接呼び出し、Ray のシリアライズ機構で IPC ハンドルを渡します。

### HTTP モード { #http-mode }

vLLM を HTTP サーバーとして動かす場合に使います。

```python
trainer_args = IPCTrainerSendWeightsArgs(
    send_mode="http",
    url="http://localhost:8000",
)

# start
base_url = "http://localhost:8000"
url = f"{base_url}/start_weight_update"
response = requests.post(url, json={}, timeout=60)
response.raise_for_status()
# send weights
IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
# finish
url = f"{base_url}/finish_weight_update"
response = requests.post(url, json={}, timeout=60)
response.raise_for_status()
```

HTTP モードでは、IPC ハンドルは pickle 化・base64 エンコードされ、JSON として `/update_weights` エンドポイントへ送られます。ワーカーは `pickle.loads` でペイロードを復元するため、vLLM サーバーは `VLLM_ALLOW_INSECURE_SERIALIZATION=1` を付けて起動する必要があります。

```python
def my_custom_sender(update_info: IPCWeightTransferUpdateInfo):
    # Custom logic to deliver update_info to vLLM
    ...

trainer_args = IPCTrainerSendWeightsArgs(
    send_mode=my_custom_sender,
)

IPCWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

設定できる項目の一覧は [`IPCTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/ipc_engine.py) を参照してください。

## 例 { #examples }

- [IPC による重み同期を使った RLHF（オフライン、Ray）](../../../examples/rl/rlhf_ipc.py) - Ray のプレースメントグループと CUDA IPC ハンドルを使い、単一 GPU 上で学習と推論を同居させる例
- [IPC による重み同期を使った RLHF（オンラインサービング、HTTP）](../../../examples/rl/rlhf_http_ipc.py) - サーバーとトレーナーが同じ GPU を共有する vLLM HTTP サーバーでの重み転送
