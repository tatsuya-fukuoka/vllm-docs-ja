# NCCL エンジン { #nccl-engine }

NCCL の重み転送エンジンは、[NCCL](https://developer.nvidia.com/nccl) のブロードキャストを使って、トレーナーから推論ワーカーへ重みを転送します。トレーナーと推論エンジンが別々の GPU で動く、**複数ノード**・**複数 GPU** の構成をサポートします。

## NCCL を使う場面 { #when-to-use-nccl }

- 学習と推論が**別々の GPU**（ノードをまたぐ場合もある）で動く
- **テンソル並列**の推論で、複数のワーカーすべてに更新後の重みが必要
- NVLink や InfiniBand を介した高帯域・低レイテンシの重み転送が必要

## 仕組み { #how-it-works }

1. トレーナーとすべての推論ワーカーが、`StatelessProcessGroup`（torch.distributed に依存しない vLLM のグループ抽象）を使って共通の NCCL プロセスグループに参加します。
2. トレーナーがすべてのワーカーへ同時に重みをブロードキャストします。各ワーカーはそれを受け取って読み込みます。
3. 任意で、**パックされたテンソルのブロードキャスト**により、複数の小さなテンソルをより大きなバッファにまとめ、ダブル / トリプルバッファリングと CUDA ストリームの重ね合わせでスループットを高められます。この実装は [NeMo-RL の packed tensor](https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/utils/packed_tensor.py) にもとづいています。

## 初期化 { #initialization }

NCCL では、プロセスグループを明示的に構築する必要があります。トレーナーと推論ワーカーは、マスターのアドレス・ポート・world size について合意しておく必要があります。

### 推論側 { #inference-side }

```python
from vllm.distributed.weight_transfer.base import WeightTransferInitRequest

# rank_offset accounts for the trainer occupying rank 0
llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,  # trainer + all inference workers
        )
    )
)
```

### トレーナー側 { #trainer-side }

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)

group = NCCLWeightTransferEngine.trainer_init(
    dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
    )
)
```

!!! note
    `trainer_init` は常にトレーナーをランク 0 に割り当てます。推論ワーカーは `rank_offset`（通常は 1）から始まります。

## 重みの送信 { #sending-weights }

```python
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

trainer_args = NCCLTrainerSendWeightsArgs(
    group=group,
    packed=True,  # use packed broadcasting for efficiency
)

NCCLWeightTransferEngine.trainer_send_weights(
    iterator=model.named_parameters(),
    trainer_args=trainer_args,
)
```

設定できる項目の一覧は [`NCCLTrainerSendWeightsArgs`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/weight_transfer/nccl_engine.py) を参照してください。

### パックされたテンソルのブロードキャスト { #packed-tensor-broadcasting }

`packed=True` にすると、ブロードキャストの前に複数の重みテンソルが大きな連続バッファにまとめられます。これにより NCCL の操作回数が減り、専用の CUDA ストリームによるダブル / トリプルバッファリングで、パック・ブロードキャスト・アンパックを重ね合わせて実行できます。

トレーナー側（`NCCLTrainerSendWeightsArgs`）と推論側（`NCCLWeightTransferUpdateInfo`）で、`packed_buffer_size_bytes` と `packed_num_buffers` の値を一致させる必要があります。

## 重みの受信（推論側） { #receiving-weights-inference-side }

推論側は、4 フェーズのプロトコルで重みの受信を進めます。
`init_weight_transfer_engine`、`start_weight_update`、`update_weights`、
`finish_weight_update` です。初期化フェーズは[上記](#initialization)のとおりで、
残りの 3 ステップは次のようになります。

```python
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

# 1. Start the weight update
llm.start_weight_update()

# 2. Receive weights (can be called multiple times for chunked transfers)
llm.update_weights(
    WeightTransferUpdateRequest(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)

# 3. Finish the weight update
llm.finish_weight_update()
```

`names`、`dtype_names`、`shapes` の各リストは、それぞれのパラメータを表します。
これらは、トレーナー側がパラメータを走査する順序と一致している必要があります。

`start_weight_update` は `update_weights` の前に、
`finish_weight_update` はすべての重みのチャンクを転送し終えた後に
呼び出す必要があります。NCCL エンジンはチェックポイント形式の重みを受け取り、
`start_weight_update` / `finish_weight_update` の内部で層単位の再読み込み処理を
自動的に適用します。

## Sparse NCCL { #sparse-nccl }

フラットインデックスの疎な重みパッチには、別のバックエンド
`WeightTransferConfig(backend="sparse_nccl")` を使います。実装は
`SparseNCCLWeightTransferEngine` です。dense のエンジンとは NCCL のプロセスグループの
初期化だけを共有し、パッチは既存のパラメータへ直接 in-place で適用されます
（層単位の再読み込みは行いません）。現在の疎版の MVP は `TP=1` と `PP=1` が
必要です。以下の例を参照してください。

## 例 { #examples }

- [NCCL による重み同期を使った RLHF（オフライン、Ray）](../../../examples/rl/rlhf_nccl.py) - トレーナーを 1 台の GPU、テンソル並列 2 の vLLM エンジンを別の 2 台で動かし、パックされた NCCL ブロードキャストで重みを転送する例
- [Sparse NCCL による重み同期を使った RLHF（オフライン、Ray）](../../../examples/rl/rlhf_sparse_nccl.py) - 2 GPU のトレーナー / 推論構成で実モデルを使い、dense と sparse の同等性を示すデモ。疎パッチは `backend="sparse_nccl"` を使い、現時点では `TP=1` と `PP=1` が必要
- [非同期の重み同期を使った RLHF（オフライン、Ray）](../../../examples/rl/rlhf_async_new_apis.py) - 実行中の一時停止、重み同期、再開、新しいモデルとの検証を伴う非同期生成の例
- [NCCL による重み同期を使った RLHF（オンラインサービング、HTTP）](../../../examples/rl/rlhf_http_nccl.py) - 稼働中の vLLM HTTP サーバーに対し、制御は HTTP、データ転送は NCCL で行う重み転送の例
