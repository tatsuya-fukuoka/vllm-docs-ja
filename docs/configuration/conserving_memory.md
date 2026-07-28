# メモリを節約する { #conserving-memory }

大きなモデルを使うと、マシンのメモリが不足（OOM）することがあります。ここでは、この問題を緩和するためのオプションを紹介します。

## テンソル並列 (TP) { #tensor-parallelism-tp }

テンソル並列（`tensor_parallel_size` オプション）を使うと、モデルを複数の GPU に分割できます。

次のコードは、モデルを 2 つの GPU に分割します。

```python
from vllm import LLM

llm = LLM(model="ibm-granite/granite-3.1-8b-instruct", tensor_parallel_size=2)
```

!!! warning
    vLLM が CUDA を正しく初期化できるよう、vLLM の初期化前に関連する関数（[torch.accelerator.set_device_index][] など）を呼び出さないでください。
    呼び出すと `RuntimeError: Cannot re-initialize CUDA in forked subprocess` のようなエラーが発生することがあります。

    使用するデバイスを制御したい場合は、代わりに環境変数 `CUDA_VISIBLE_DEVICES` を設定してください。

!!! note
    テンソル並列を有効にすると、各プロセスがモデル全体を読み込んでから分割するため、ディスクからの読み込み時間がさらに長くなります（テンソル並列のサイズに比例します）。

    [examples/features/sharded_state/load_sharded_state_offline.py](../../examples/features/sharded_state/load_sharded_state_offline.py) を使うと、モデルのチェックポイントをシャード化されたチェックポイントに変換できます。変換には時間がかかりますが、以降はシャード化されたチェックポイントをずっと高速に読み込めます。モデルの読み込み時間は、テンソル並列のサイズによらず一定になります。

## 量子化 { #quantization }

量子化されたモデルは、精度と引き換えにメモリ使用量が少なくなります。

静的に量子化されたモデルは HF Hub からダウンロードでき（よく使われるものは [Red Hat AI](https://huggingface.co/RedHatAI) にあります）、追加の設定なしにそのまま利用できます。

動的量子化も `quantization` オプションでサポートされています。詳細は[こちら](../features/quantization/README.md)を参照してください。

## コンテキスト長とバッチサイズ { #context-length-and-batch-size }

モデルのコンテキスト長（`max_model_len` オプション）と最大バッチサイズ（`max_num_seqs` オプション）を制限することで、メモリ使用量をさらに減らせます。

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct", max_model_len=2048, max_num_seqs=2)
```

## CUDA グラフを減らす { #reduce-cuda-graphs }

vLLM は既定で CUDA グラフを使ってモデル推論を最適化しますが、これは GPU の追加メモリを消費します。

`compilation_config` を調整することで、推論速度とメモリ使用量のバランスを取れます。

??? code

    ```python
    from vllm import LLM
    from vllm.config import CompilationConfig, CompilationMode

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            # By default, it goes up to max_num_seqs
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
    )
    ```

`enforce_eager` フラグを使うと、グラフのキャプチャを完全に無効化できます。

```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=True)
```

## キャッシュサイズを調整する { #adjust-cache-size }

CPU の RAM が不足する場合は、次のオプションを試してください。

- （マルチモーダルモデルのみ）エンジン引数 `mm_processor_cache_gb` でマルチモーダルキャッシュのサイズを設定できます（既定は 4 GiB）。
- （CPU バックエンドのみ）環境変数 `VLLM_CPU_KVCACHE_SPACE` で KV キャッシュのサイズを設定できます（既定は 4 GiB）。

## マルチモーダル入力の上限 { #multi-modal-input-limits }

プロンプトごとのマルチモーダル項目数を少なく制限すると、モデルのメモリ使用量を減らせます。

```python
from vllm import LLM

# Accept up to 3 images and 1 video per prompt
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 3, "video": 1},
)
```

さらに踏み込んで、上限を 0 にすることで使わないモダリティを完全に無効化できます。
たとえば、アプリケーションが画像入力しか受け付けないのであれば、動画用にメモリを確保する必要はありません。

```python
from vllm import LLM

# Accept any number of images but no videos
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"video": 0},
)
```

マルチモーダルモデルをテキスト専用の推論で動かすこともできます。

```python
from vllm import LLM

# Don't accept images. Just text.
llm = LLM(
    model="google/gemma-3-27b-it",
    limit_mm_per_prompt={"image": 0},
)
```

### 設定可能なオプション { #configurable-options }

`limit_mm_per_prompt` は、モダリティごとに詳細なオプションを指定することもできます。この形式では `count` を指定したうえで、vLLM がマルチモーダル入力のメモリをどうプロファイリング・確保するかを制御するサイズのヒントを任意で与えられます。これにより、モデルの絶対的な上限ではなく、実際に想定するメディアに合わせてメモリを調整できます。

モダリティごとに設定できるオプション:

- `image`: `{"count": int, "width": int, "height": int}`
- `video`: `{"count": int, "num_frames": int, "width": int, "height": int}`
- `audio`: `{"count": int, "length": int}`

詳細は [`ImageDummyOptions`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/multimodal/#vllm.config.multimodal.ImageDummyOptions)、[`VideoDummyOptions`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/multimodal/#vllm.config.multimodal.VideoDummyOptions)、[`AudioDummyOptions`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/multimodal/#vllm.config.multimodal.AudioDummyOptions) を参照してください。

例:

```python
from vllm import LLM

# Up to 5 images per prompt, profile with 512x512.
# Up to 1 video per prompt, profile with 32 frames at 640x640.
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={
        "image": {"count": 5, "width": 512, "height": 512},
        "video": {"count": 1, "num_frames": 32, "width": 640, "height": 640},
    },
)
```

後方互換性のため、整数を渡した場合は従来どおり動作し、`{"count": <int>}` として解釈されます。例:

- `limit_mm_per_prompt={"image": 5}` は `limit_mm_per_prompt={"image": {"count": 5}}` と同じです
- 形式を混在させることもできます: `limit_mm_per_prompt={"image": 5, "video": {"count": 1, "num_frames": 32, "width": 640, "height": 640}}`

!!! note
    - サイズのヒントはメモリのプロファイリングにのみ影響します。確保するアクティベーションのサイズを計算するためのダミー入力の形を決めるものであり、推論時に入力が実際にどう処理されるかを変えるものではありません。
    - ヒントがモデルの受け入れ可能な範囲を超えている場合、vLLM はモデルの実効的な最大値に丸め、警告を出力することがあります。

!!! warning
    これらのサイズのヒントは、現時点ではアクティベーションメモリのプロファイリングにのみ影響します。エンコーダーキャッシュのサイズは実行時の実際の入力によって決まり、これらのヒントでは制限されません。

## マルチモーダルプロセッサの引数 { #multi-modal-processor-arguments }

一部のモデルでは、マルチモーダルプロセッサの引数を調整して処理後のマルチモーダル入力のサイズを小さくし、メモリを節約できます。

例:

```python
from vllm import LLM

# Available for Qwen2-VL series models
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_kwargs={"max_pixels": 768 * 768},  # Default is 1280 * 28 * 28
)

# Available for InternVL series models
llm = LLM(
    model="OpenGVLab/InternVL2-2B",
    mm_processor_kwargs={"max_dynamic_patch": 4},  # Default is 12
)
```
