# 層単位の（再）読み込みとは { #what-is-layerwise-reloading }

層単位の再読み込み (layerwise reloading) は、CUDA グラフなどの実行時アーティファクトの再コンパイルを引き起こさずに、既存の重みの格納先へ新しい重みデータを読み込むための仕組みです。[QeRL](https://arxiv.org/pdf/2510.11696) 形式の事後学習フローを実現するために使われます。QeRL では、全精度のトレーナーの重みを量子化して対象の vLLM インスタンスへ読み込み、高速かつ探索性の高いロールアウトを行います。中心的な実装は [layerwise.py](../../vllm/model_executor/model_loader/reload/layerwise.py) にあります。

![Layerwise](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/training/layerwise.png)

## QeRL のための層単位の再読み込み { #layerwise-reloading-for-qerl }

既存の重みの格納先へ新しい重みを読み込むには、重みに対して次の処理が必要です。

- 転送: トレーナーのモデルから対象のノード / デバイスへ重みを転送する
- 融合: qkv や gate_up など、分割された重みを融合する
- 処理: 通常はオンライン量子化や、カーネル固有のパディング・ストライドの調整を指す
- 分割: 選択した並列化戦略に従って重みを分割する
- コピー: 既存の重みの格納先へ重みをコピーする

層単位の再読み込みは、次の手順でこれを実現します。

1. トレーナーから対象へ重みを**転送**します（[重み転送](weight_transfer/README.md)を参照）
2. `model.load_weights` で重みを読み込みます。この過程で**分割**と**融合**が行われます
3. ある層の重みがすべて読み込まれた時点で、オンラインに**処理**されます
4. 既存の重みの格納先へ**コピー**されます

実装の詳細は[低レベルの `layerwise` API](#low-level-layerwise-api) を参照してください。

## オンライン量子化を伴う層単位の読み込み { #layerwise-loading-with-online-quantization }

オンライン量子化とは、利用者が全精度の重みを与え、それらをモデルへ読み込む際にその場で量子化することを指します。層単位の再読み込みの仕組みでは、オンライン量子化を**処理**のステップとして扱い、初回の読み込み時と再読み込み時の両方でオンラインに処理します。オンライン量子化の実装は通常、次のような形になります。

```python
class Fp8PerTensorOnlineLinearMethod(LinearMethodBase):
    """Online version of FP8 per-tensor quantization which loads a full
    precision checkpoint and quantizes weights during loading."""

    uses_meta_device: bool = True

    def create_weights(self, layer: torch.nn.Module, ...):
        # weight is materialized and processed during loading
        layer.weight = ModelWeightParameter(
            data=torch.empty(..., device="meta"),
            weight_loader=weight_loader,
        )

        # set up online processing
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        layer.weight, layer.weight_scale = ops.scaled_fp8_quant(layer.weight)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True
```

## 使用例 { #example-usages }

### 高レベルの重み転送 API { #high-level-weight-transfer-api }

層単位の再読み込みは、事後学習の重み転送の仕組みと統合されています。重み転送と組み合わせて層単位の再読み込みを使うには、[こちら](../../examples/rl/)の例を参照してください。チェックポイント形式の重み転送エンジン（NCCL と IPC のバックエンドなど）は、`start_weight_update` / `finish_weight_update` のライフサイクルの中で層単位の再読み込みを自動的に実行します。

### 中レベルの `reload_weights` API { #mid-level-reload_weights-api }

層単位の再読み込みは `reload_weights` API からも利用できます。次のコードで呼び出せます。

```python
from vllm import LLM

llm = LLM("Qwen/Qwen3-0.6B")
llm.collective_rpc("reload_weights")
```

このインターフェイスでは `weights_path` も指定でき、読み込み元のチェックポイントのパスを選べます。

```python
from vllm import LLM

# fine tuned model checkpoints for testing
mul_path = "inference-optimization/Qwen3-0.6B-debug-multiply"
add_path = "inference-optimization/Qwen3-0.6B-debug-add"

llm = LLM("Qwen/Qwen3-0.6B")
llm.collective_rpc("reload_weights", kwargs={"weights_path": mul_path})
llm.generate("3 4 = ")  # 12

llm.collective_rpc("reload_weights", kwargs={"weights_path": add_path})
llm.generate("3 4 = ")  # 7
```

さらに、`weights_iterator` を直接渡すこともできます。このイテレータは遅延評価でも即時評価でも構いません。

```python
from vllm import LLM

weights_iterator = [("q_proj", ...), ("k_proj", ...), ...]

llm = LLM("Qwen/Qwen3-0.6B")
llm.collective_rpc("reload_weights", kwargs={"weights_iterator": weights_iterator})
```

### 低レベルの `layerwise` API { #low-level-layerwise-api }

[layerwise.py](../../vllm/model_executor/model_loader/reload/layerwise.py) は、ライフサイクルを実行するために次の関数を実装しています。

| 関数 | 目的 | 量子化済みの再読み込み | オンライン量子化 |
| - | - | - | - |
| `record_metadata_for_reloading` | メタデバイス上で層を復元できるよう、テンソルのメタデータを記録する | `BaseModelLoader` が呼び出す | `BaseModelLoader` が呼び出す |
| `restore_layer_on_meta` | 再読み込みの開始時に、層をモデルの形式へ復元する | `initialize_layerwise_reload` が呼び出す | 呼び出されない。オンライン量子化された重みは `...OnlineLinearMethod.create_weights` により最初からメタデバイス上にある |
| `initialize_online_processing` | 層のすべての重みが読み込まれるまで重みをバッファする `online_process_loader` で重みローダーをラップする | `initialize_layerwise_reload` が呼び出す | `...OnlineLinearMethod.create_weights` が呼び出す |
| `_layerwise_process` | すべての重みが読み込まれた時点で層を処理する | 読み込み中に `online_process_loader` が呼び出す | 読み込み中に `online_process_loader` が呼び出す |
| `_copy_and_restore_kernel_tensors` | コンパイル済みの CUDA グラフなどに反映されるよう、処理後の重みを元のテンソルの位置へコピーする | `process_weights_after_loading` の後に `_layerwise_process` が呼び出す | 呼び出されない。まだコンパイル済みの CUDA グラフが存在しないため |
| `finalize_layerwise_processing` | すべての重みが読み込まれなかった層（Attention の重みやパディングを含む重みなど）を拾う | `BaseModelLoader` が呼び出す | `BaseModelLoader` が呼び出す |

`initialize_layerwise_reload` を呼び、重みを読み込み、`finalize_layerwise_processing` を呼ぶことで、このライフサイクルに直接組み込めます。

```python
from vllm import LLM
from vllm.model_executor.model_loader.reload import initialize_layerwise_reload, finalize_layerwise_processing

llm = LLM("Qwen/Qwen3-0.6B")

# this model path requires `VLLM_ENABLE_V1_MULTIPROCESSING=0` and is not stable
model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.get_model()

# layerwise reload
initialize_layerwise_reload(model)
model.load_weights(...)
finalize_layerwise_processing(model, llm.model_config)
```

## メモリ使用量が過大になる場合のトラブルシューティング { #troubleshooting-excessive-memory-usage }

層単位の再読み込みでは、重みをモデルへ読み込みながら段階的に処理できます。この仕組みは、ある層のすべての重みが読み込まれるまで、その層の重みをデバイス上にバッファすることに依存しています。ただしオフロードを行わない場合、重みが順不同で読み込まれると、どうしてもバッファが過大になります。

そのため、モデルへ再読み込みする際の重みの順序に注意する必要があります。重みは「順番どおり」に、つまり次の層の重みを読み始める前に各層の重みを読み終える形で読み込むべきです。「順不同」の読み込みでは、他の層の重みを読み込んでいる間もある層の重みがバッファに残り、メモリ使用量が過大になります。以下の例では、q_proj・k_proj・v_proj・up_proj が同時にバッファされており、up_proj を q_proj・k_proj・v_proj の後に読み込む場合よりメモリを消費しています。

| 正しい読み込み | 誤った読み込み |
| - | - |
| ![Layerwise](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/training/layerwise_good_loading.png) | ![Layerwise](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/training/layerwise_bad_loading.png) |

重みが順不同で読み込まれると、次のような警告が表示されます。

```console
WARNING [layerwise.py:198] Allocating 28.5 MB of device memory to buffers to load ["QKVParallelLinear", "MergedColumnParallelLinear"] layers. This extra memory usage can be avoided by ordering weights by their parent layer when reloading.
```
