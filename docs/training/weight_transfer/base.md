# 基底クラスと独自エンジン { #base-class-and-custom-engines }

重み転送の仕組みは、vLLM のワーカー基盤と転送バックエンドの間の取り決めを定義する抽象基底クラスの上に構築されています。`WeightTransferEngine` を継承し、`WeightTransferEngineFactory` に登録することで独自のバックエンドを実装できます。

## WeightTransferEngine { #weighttransferengine }

`WeightTransferEngine` は、2 つのデータクラス型をパラメータに取るジェネリックな抽象クラスです。

- **`TInitInfo`**（`WeightTransferInitInfo` を継承）: バックエンド固有の初期化パラメータ。
- **`TUpdateInfo`**（`WeightTransferUpdateInfo` を継承）: バックエンド固有の重み更新メタデータ。

### 抽象メソッド { #abstract-methods }

サブクラスは次のメソッドを実装する必要があります。

| メソッド | 側 | 説明 |
| ------ | ---- | ----------- |
| `init_transfer_engine(init_info)` | 推論 | 各推論ワーカーで通信路を初期化する |
| `start_weight_update()` | 推論 | 更新の準備をする（層単位の再読み込みの開始など）。in-place なエンジンでは何もしない |
| `finish_weight_update()` | 推論 | 更新を確定する（層単位の再読み込みの終了処理など）。in-place なエンジンでは何もしない |
| `receive_weights(update_info)` | 推論 | 重みを受け取り `self.model` に読み込む |
| `shutdown()` | 推論 | リソースを解放する |
| `trainer_send_weights(iterator, trainer_args)` | トレーナー | トレーナーのプロセスから重みを送る静的メソッド |

基底クラスは次の 2 つのメソッドを提供します。

1. `__init__`: エンジンは `config`（`WeightTransferConfig`）、`vllm_config`（`VllmConfig`）、`device`（`torch.device`）、`model`（`nn.Module`）を受け取ります。  
2. `update_weights(update_info_dict)`: `receive_weights` の薄いラッパーです。辞書を
指定されたデータ型に変換し、`receive_weights` を呼び出してデバイスを同期します。サブクラスは `receive_weights` を実装します。

### リクエストのクラス { #request-classes }

API レベルのリクエストクラスは、素の辞書を使ったバックエンド非依存のシリアライズを提供します。エンジンの `parse_init_info` と `parse_update_info` が、これらの辞書を型付きのデータクラスに変換します。

```python
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)

# Init request (dict is converted to backend-specific TInitInfo)
init_request = WeightTransferInitRequest(
    init_info={"master_address": "10.0.0.1", "master_port": 29500, ...}
)

# Update request (dict is converted to backend-specific TUpdateInfo)
update_request = WeightTransferUpdateRequest(
    update_info={"names": [...], "dtype_names": [...], "shapes": [...]}
)
```

LLM / API の層では、投機的デコーディングのドラフトモデルを対象にするために、
`start_weight_update()` の代わりに `start_draft_weight_update()` を呼び出します。
`update_weights` と `finish_weight_update` は同じです。

### WeightTransferUpdateInfo { #weighttransferupdateinfo }

基底の `WeightTransferUpdateInfo` は、バックエンド固有の更新情報のためのマーカークラスです。

```python
@dataclass
class WeightTransferUpdateInfo(ABC):
    pass
```

## 独自エンジンの実装 { #implementing-a-custom-engine }

独自の重み転送バックエンドを作るには次のようにします。

### 1. 情報のデータクラスを定義する { #1-define-info-dataclasses }

```python
from dataclasses import dataclass
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

@dataclass
class MyInitInfo(WeightTransferInitInfo):
    endpoint: str
    token: str

@dataclass
class MyUpdateInfo(WeightTransferUpdateInfo):
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    # Add custom fields as needed
```

### 2. エンジンを実装する { #2-implement-the-engine }

```python
from collections.abc import Iterator
from typing import Any
import torch

class MyWeightTransferEngine(WeightTransferEngine[MyInitInfo, MyUpdateInfo]):
    init_info_cls = MyInitInfo
    update_info_cls = MyUpdateInfo

    def init_transfer_engine(self, init_info: MyInitInfo) -> None:
        # Set up connection to trainer using init_info.endpoint, etc.
        ...

    def start_weight_update(self) -> None:
        # Checkpoint-format engines: run initialize_layerwise_reload(self.model).
        # In-place engines: no-op
        ...

    def finish_weight_update(self) -> None:
        # Checkpoint-format engines: run finalize_layerwise_reload(...).
        # In-place engines: no-op
        ...

    def receive_weights(self, update_info: MyUpdateInfo) -> None:
        weights = []
        for name, dtype_name, shape in zip(
            update_info.names, update_info.dtype_names, update_info.shapes
        ):
            dtype = getattr(torch, dtype_name)
            weight = self._fetch_weight(name, shape, dtype)
            weights.append((name, weight))
        self.model.load_weights(weights)

    def shutdown(self) -> None:
        # Clean up resources
        ...

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any],
    ) -> None:
        # Send weights from the trainer process
        for name, tensor in iterator:
            # Send tensor via custom transport
            ...
```

### 3. ファクトリに登録する { #3-register-with-the-factory }

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# Option 1: Lazy loading (recommended for built-in engines)
WeightTransferEngineFactory.register_engine(
    "my_backend",
    "my_package.my_module",
    "MyWeightTransferEngine",
)

# Option 2: Direct class registration
WeightTransferEngineFactory.register_engine(
    "my_backend",
    MyWeightTransferEngine,
)
```

登録すると、利用者は `WeightTransferConfig(backend="my_backend")` でそのバックエンドを選べます。

## WeightTransferEngineFactory { #weighttransferenginefactory }

ファクトリはレジストリのパターンと遅延読み込みを使います。組み込みのエンジン（`nccl`、`ipc`、`sparse_nccl`）は import 時に登録されますが、そのモジュールが実際に読み込まれるのは、そのバックエンドが要求されたときだけです。これにより、不要なときに NCCL のコミュニケータのような重い依存を import せずに済みます。

```python
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

# Create an engine from config
engine = WeightTransferEngineFactory.create_engine(
    config=weight_transfer_config,
    vllm_config=vllm_config,
    device=device,
    model=model,
)
```
