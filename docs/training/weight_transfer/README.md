# 重み転送 { #weight-transfer }

vLLM は、強化学習 (RL) のワークフローにおいて、学習プロセスから推論エンジンへモデルの重みを同期するための差し替え可能な重み転送の仕組みを提供します。これは RLHF、GRPO をはじめとするオンライン RL の手法に不可欠です。

## アーキテクチャ { #architecture }

重み転送の仕組みは、差し替え可能なバックエンド設計のもとで **4 つのフェーズからなるプロトコル**に従います。

1. **初期化** (`init_weight_transfer_engine`): トレーナーと推論ワーカーの間の通信路を確立します。学習ループの開始前に 1 回だけ呼び出します。
2. **開始** (`start_weight_update`): 重み更新に向けて推論エンジンを準備します。
3. **重みの更新** (`update_weights`): 更新後の重みをトレーナーから推論エンジンへ転送します。分割転送などのため、複数回呼び出されることがあります。
4. **終了** (`finish_weight_update`): 重み更新を確定します（チェックポイント形式の重みに対する後処理の実行など）。すべての重みを転送した後に 1 回だけ呼び出します。

## 利用できるバックエンド { #available-backends }

| バックエンド | 転送方式 | ユースケース |
| ------- | --------- | -------- |
| [NCCL](nccl.md) | NCCL のブロードキャスト | 学習と推論で GPU を分ける場合 |
| [IPC](ipc.md) | CUDA IPC ハンドル | 学習と推論を同じ GPU に同居させる場合 |
| [sparse_nccl](nccl.md#sparse-nccl) | NCCL のブロードキャスト | フラットインデックスの疎な重みパッチ (TP=1/PP=1) |

## 設定 { #configuration }

重み転送のバックエンドは `WeightTransferConfig` で指定します。バックエンドによって、どのエンジンが重み同期を担当するかが決まります。

### プログラムから指定する（オフライン推論） { #programmatic-offline-inference }

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(
    model="my-model",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),  # or "ipc"
)
```

### CLI から指定する（オンラインサービング） { #cli-online-serving }

```bash
vllm serve my-model \
    --weight-transfer-config '{"backend": "nccl"}'
```

`backend` フィールドには `"nccl"`（既定）、`"ipc"`、`"sparse_nccl"` を指定できます。

## API エンドポイント { #api-endpoints }

vLLM を HTTP サーバーとして動かす場合、重み転送のために次のエンドポイントを利用できます。

| エンドポイント | メソッド | 説明 |
| -------- | ------ | ----------- |
| `/init_weight_transfer_engine` | POST | バックエンド固有の情報で重み転送エンジンを初期化する |
| `/start_weight_update` | POST | 重み更新を開始する |
| `/update_weights` | POST | バックエンド固有のメタデータとともに重みのバッチを転送する |
| `/finish_weight_update` | POST | 重み更新を終了し、後処理を実行する |
| `/pause` | POST | 重み同期の前に生成を一時停止し、実行中のリクエストを処理する |
| `/resume` | POST | 重み同期の後に生成を再開する |
| `/get_world_size` | GET | 推論ワーカーの数を取得する（NCCL の world size の計算に便利） |

!!! note
    HTTP の重み転送エンドポイントを使うには `VLLM_SERVER_DEV_MODE=1` の設定が必要です。

## トレーナー側の API { #trainer-side-api }

どちらのバックエンドも、重みを送るためにトレーナーが呼び出す静的メソッドを提供します。基本的な流れは次のとおりです。

```python
# 1. Initialize the transfer engine (backend-specific)
EngineClass.trainer_init(init_info)

# 2. Start weight update on inference side
llm.start_weight_update()

# 3. Send weights to inference workers
EngineClass.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=backend_specific_args,
)

# 4. Finish weight update on inference side
llm.finish_weight_update()
```

バックエンドごとのトレーナー API と完全な例は [NCCL](nccl.md) と [IPC](ipc.md) のページを参照してください。

## 仕組みを拡張する { #extending-the-system }

重み転送の仕組みは拡張できるよう設計されています。`WeightTransferEngine` を継承してファクトリに登録することで、独自のバックエンドを実装できます。詳細は[基底クラス](base.md)のページを参照してください。
