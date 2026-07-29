# Vision Encoder（ViT）の CUDA Graphs { #vision-encoder-vit-cuda-graphs }

vLLM の [CUDA Graphs](cuda_graphs.md) の仕組みは、主に**デコーダ**（言語モデル）の forward pass を対象としています。vLLM は、デコーダとは独立に**エンコーダ**（vision transformer）の forward pass を CUDA Graphs としてキャプチャすることもサポートしています。これは <https://github.com/vllm-project/vllm/pull/35963> にもとづいています。

2 タワー構成の vision エンコーダ（動的タイリングを伴う DeepSeek-OCR の SAM + CLIP など）向けには、**dual-path graph** モードが 2 つの独立した CUDA graph の集合（グローバル画像パス用とローカルパッチパス用）をキャプチャし、パスごとに独立した budget の選択と部分的な eager フォールバックを可能にします。これは <https://github.com/vllm-project/vllm/pull/43586> にもとづいています。

!!! note
    エンコーダの CUDA Graphs はデコーダの CUDA Graphs と直交しており、両方を同時に有効にできます。エンコーダのグラフは vision エンコーダの実行（Qwen3-VL の ViT など）をキャプチャし、デコーダのグラフは [CUDA Graphs の設計ドキュメント](cuda_graphs.md)で説明したとおり言語モデルの実行をキャプチャします。

## 動機 { #motivation }

vision エンコーダの推論では、ホスト側で CUDA カーネルの起動オーバーヘッドが発生します。バッチサイズが小さい場合や画像サイズが小さい場合、このオーバーヘッドはより顕著になります。

エンコーダの CUDA Graphs は、モデル初期化時にエンコーダの forward pass 全体を複数のトークン budget レベルであらかじめキャプチャし、実行時に適切なグラフを再生することで、このオーバーヘッドを解消します。

DeepSeek-OCR（動的タイリングを伴う SAM + CLIP）のような 2 タワー構成の vision エンコーダでは、グローバル画像パスとローカルパッチパスが独立したトークンのプロファイルを持ちます（グローバル画像 1 枚あたり 272 トークン、ローカルパッチ 1 個あたり 100 トークン）。両方のパスを 1 つの一枚岩のグラフとしてキャプチャすると、パッキングの効率が大きく下がってしまいます。dual-path graph モードは各パスを別々の budget の集合としてキャプチャし、マネージャがパスごとに独立してパッキングと再生を行えるようにします。

## 設計 { #design }

エンコーダの CUDA Graph システムは、[`EncoderCudaGraphManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/encoder_cudagraph/#vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager) が管理する **budget ベースのキャプチャ / 再生**戦略を採用しています。システムは次の中核コンポーネントから成ります。

* [`EncoderCudaGraphManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/encoder_cudagraph/#vllm.v1.worker.encoder_cudagraph.EncoderCudaGraphManager): エンコーダ CUDA Graphs のキャプチャ、再生、貪欲なパッキング、データ並列実行を統括します。
* [`SupportsEncoderCudaGraph`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph): モデルがエンコーダ CUDA Graphs に対応するために実装する、実行時に検査可能なプロトコルです。
* [`EncoderItemSpec`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/encoder_cudagraph_defs/#vllm.v1.worker.encoder_cudagraph_defs.EncoderItemSpec): 1 つのエンコーダ入力項目（画像または動画）を、その入力サイズと出力トークン数とともに記述します。
* [`BudgetGraphMetadata`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/encoder_cudagraph/#vllm.v1.worker.encoder_cudagraph.BudgetGraphMetadata): 1 つのトークン budget レベルについて、キャプチャした CUDA Graph と関連する入出力バッファを保持します。

### budget ベースのグラフキャプチャ { #budget-based-graph-capture }

異なる**トークン budget** のレベル（`[2048, 4096, 8192, 13824]` など）で複数の CUDA Graphs をあらかじめキャプチャします。各 budget は固定のトークン容量を定義し、すべての budget が同じ最大バッチサイズ（画像枚数）を共有します。各レベルの `BudgetGraphMetadata` は、グラフとともに、あらかじめ確保した入力・メタデータ・出力のバッファを保持します。

```python
@dataclass
class BudgetGraphMetadata:
    token_budget: int
    max_batch_size: int
    max_frames_per_batch: int
    graph: torch.cuda.CUDAGraph
    input_buffers: dict[str, torch.Tensor]  # e.g. pixel_values, embeddings, seq metadata
    output_buffer: torch.Tensor      # encoder hidden states
```

budget は `get_encoder_cudagraph_budget_range()` によりモデルが提供する範囲から 2 のべき乗のレベルとして自動生成され、最大 budget は 2 のべき乗の境界に一致しない場合でも常に含まれます。budget は `CompilationConfig` の `encoder_cudagraph_token_budgets` でユーザーが明示的に指定することもできます。

`EncoderCudaGraphConfig.enable_dual_path_graph` が `True` の場合、マネージャは 2 つの独立した budget のリスト（`global_token_per_image` の倍数からなる `global_token_budgets` と、`local_token_per_patch` の倍数からなる `local_token_budgets`）を生成し、キャプチャしたグラフをそれぞれ `budget_graphs["global"]` と `budget_graphs["local"]` に格納します。

### 実行時の貪欲なビンパッキング { #greedy-bin-packing-at-runtime }

画像のバッチが届くと、マネージャは出力トークン数の小さい順に画像を並べ替え、**最大の**トークン budget と最大バッチサイズを超えない範囲で、各サブバッチにできるだけ多くの画像を貪欲に詰め込みます。サブバッチが確定すると（次の画像を入れるといずれかの制約を超える場合）、マネージャはそのサブバッチの合計トークン数に収まる**最小の** budget を見つけ、対応する CUDA Graph を再生します。これをバッチが尽きるまで繰り返します。すべての budget を超える画像は eager 実行にフォールバックします。

dual-path のモデルでは、マネージャは `_execute_local_dual_path()` へ処理を振り分けます。これはパッキング時にグローバルとローカル双方のトークン budget を同時に制約します（[Dual-Path のグラフキャプチャ](#dual-path-graph-capture)を参照）。

グラフ再生ごとの処理は次のとおりです。

1. `prepare_encoder_cudagraph_replay_buffers()` を呼び出し、実際のバッチ入力から（`pixel_values` や事前計算済みメタデータを含む）バッファの値を計算します。
2. あらかじめ確保した `input_buffers` をゼロで埋め、再生用の値をスライスコピーで書き込みます。
3. CUDA Graph を再生します。
4. `output_buffer` から出力を clone します（バッファは再生ごとに再利用されるため clone が必要です）。

### Dual-Path のグラフキャプチャ { #dual-path-graph-capture }

2 タワー構成の vision エンコーダ（DeepSeek-OCR など）では、`EncoderCudaGraphConfig` が `enable_dual_path_graph=True` を設定し、`global_token_per_image` / `local_token_per_patch` を提供します。マネージャは 2 つの独立した CUDA graph の集合（**グローバル**画像パス用と**ローカル**パッチパス用）をキャプチャし、それぞれ `budget_graphs["global"]` と `budget_graphs["local"]` に格納します。

**budget の生成。** 2 つの別々の budget リストが生成されます。

* `global_token_budgets` — `global_token_per_image` の 2 のべき乗倍（DeepSeek-OCR では `[272, 544, 1088, 2176, 4352, 8704, 13824]` など）。
* `local_token_budgets` — `local_token_per_patch` の 2 のべき乗倍（DeepSeek-OCR では `[0, 100, 200, 400, 800, 1600, 3200, 6400, 12800]` など）。ローカルパッチを持たない画像（640×640 以下で、グローバル特徴のみを生成する画像）に対応するため、budget `0` が常に含まれます。

いずれのリストも同じ `max_budget` で上限が設けられます。

**dual-path の貪欲なパッキング。** 各 `EncoderItemSpec` は `global_output_tokens`（画像ごとに一定）と `local_output_tokens`（パッチ数に比例）の両方を提供します。dual-path のパッキングアルゴリズムは、両方の budget を同時に制約します。

* 画像を合計出力トークン数（グローバル + ローカル）の小さい順に並べ替えます。
* 貪欲に画像を詰め込みます。画像が現在のサブバッチに追加されるのは、累積グローバルトークンが `max_global_budget` 以下**かつ**累積ローカルトークンが `max_local_budget` 以下で、画像枚数が `max_batch_size` 以下の場合のみです。
* いずれかの制約を超えそうになった時点でサブバッチを確定し、各パスについて**独立に**収まる最小の budget を探します。
* すべての画像を詰め終わるまで繰り返します。

**部分的なグラフのフォールバック。** パッキング後、各サブバッチは次の 4 つの実行シナリオのいずれかになります。

| グローバル budget | ローカル budget | 実行 |
| :---: | :---: | --- |
| あり | あり | 両方のパスで CUDA graph を再生 |
| あり | `None` | グローバルはグラフ再生、ローカルパスはスキップ（パッチなし） |
| `None` | あり | グローバルは eager フォールバック、ローカルはグラフ再生 |
| `None` | `None` | 両方のパスが eager 実行にフォールバック |

なお、ローカル側で budget `0` のグラフが実際に再生されることはありません。これはローカルパッチの処理を完全にスキップすべきことを示すものです。

**パスごとのバッファキー。** グローバルとローカルのパスは異なるバッファキーを使います。DeepSeek-OCR では、グローバルパスが `pixel_values`（画像全体、形状 `[B, 3, 1280, 1280]`）を使い、ローカルパスが `images_crop`（パッチ、形状 `[P, 3, 1024, 1024]`）を使います。マネージャは共有の `buffer_keys` リストではなく、キャプチャされた各グラフ自身の `input_buffers.keys()` を走査するため、両方のパスで異なるバッファを使えます。

**後処理。** `postprocess_encoder_output` メソッドは、ローカルパスのエンコーダ出力を含む `local_output` パラメータ（テンソルまたは `None`）を受け取ります。グローバル特徴とローカル特徴を組み立てて画像ごとの最終的な埋め込みにするのはモデルの責任です。DeepSeek-OCR の場合、グローバル出力を `[B, 272, n_embed]` に、ローカル出力を `[P, 100, n_embed]` に整形し、改行トークンを含むパッチグリッドを組み立て、画像ごとに `[patches_grid, global, view_separator]` を連結することを意味します。

!!! note
    dual-path の設計により、CUDA graph の部分的な適用が可能になります。一方のパスがヒットし、他方が eager にフォールバックすることができます。これにより、タイル分割されていない画像でゼロ埋めされたパッチバッファに無駄な計算を行うことを避け、画像ごとに変化する `crop_shape` によるグラフの無効化も回避できます。

### データ並列のサポート { #data-parallel-support }

`mm_encoder_tp_mode="data"` の場合、マネージャは `get_load_balance_assignment` による負荷分散された割り当てを使って画像を TP ランク間に分配し、各ランクでローカルに実行したうえで、`tensor_model_parallel_all_gather` により元の順序で結果を集約します。

### 動画推論のサポート { #video-inference-support }

<https://github.com/vllm-project/vllm/pull/35963>（画像推論向けの ViT フル CUDA graph サポート）に続き、<https://github.com/vllm-project/vllm/pull/38061> はエンコーダの CUDA graph フレームワークを拡張し、Qwen3-VL の動画推論をサポートしました。以前は、CUDA graph のキャプチャ / 再生の経路は画像入力（`pixel_values` + `image_grid_thw`）しか扱えませんでした。動画入力は異なるキー（`pixel_values_videos` + `video_grid_thw`）を使い、各動画項目が複数フレーム（`T` 個の attention シーケンス）を持つため、より大きな `cu_seqlens` バッファを必要とします。この PR はプロトコルとマネージャを一般化し、単一の共有グラフマネージャで両方のモダリティを扱えるようにしました。

!!! note
    EVS（Efficient Video Sampling）による枝刈りが有効な場合、動画の CUDA graph は自動的に無効になります。EVS はトークン数をデータ依存にするため、CUDA graph のキャプチャと両立しないためです。

    プロンプトごとの入力の混在（画像 + 動画）も現在サポートされています。

## `SupportsEncoderCudaGraph` によるモデルの統合 { #model-integration-via-supportsencodercudagraph }

モデルは [`SupportsEncoderCudaGraph`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsEncoderCudaGraph) プロトコルを実装することで、エンコーダの CUDA Graphs に対応します。このプロトコルはモデル固有のロジックをすべてカプセル化し、マネージャがモデルに依存しないようにします。プロトコルは次のメソッドを定義します。

* `get_encoder_cudagraph_config()` — 静的な設定（サポートするモダリティ、バッファキー、出力の hidden size、パディングのロジック、動画あたりの最大フレーム数）を返します。
* `get_encoder_cudagraph_budget_range(vllm_config)` — トークン budget の自動推定のための `(min_budget, max_budget)` を返します。
* `get_encoder_cudagraph_item_specs(mm_kwargs)` — 各項目について入力サイズ、合計出力トークン数（`output_tokens`）、および dual-path モデルの場合は任意でパスごとのトークン数（`global_output_tokens`、`local_output_tokens`）を記述する `list[EncoderItemSpec]` を返します。
* `select_encoder_cudagraph_items(mm_kwargs, indices)` — インデックスにより項目のサブバッチを抽出します。貪欲なパッキングと DP のシャーディングで使われます。
* `prepare_encoder_cudagraph_capture_inputs(..., path="default")` — グラフキャプチャ用のダミー入力を作成します。`path` パラメータ（`"global"` または `"local"`）は、どのパス向けのダミー入力を生成するかをモデルに伝えます。グラフに記録されるすべてのバッファを含む単一の `values: dict[str, torch.Tensor]` を持つ `EncoderCudaGraphCaptureInputs` を返します。
* `prepare_encoder_cudagraph_replay_buffers(mm_kwargs, max_batch_size, max_frames_per_batch, path="default")` — 実際のバッチ入力からバッファの値を計算します。`path` パラメータは `mm_kwargs` から抽出するモダリティのキーを選択します。キーがキャプチャ済みグラフの `input_buffers.keys()` と一致する `values` 辞書を持つ `EncoderCudaGraphReplayBuffers` を返します。
* `encoder_cudagraph_forward(inputs: dict[str, torch.Tensor], path="default")` — 固定形状の入力テンソル（キャプチャされた `values` 辞書）のみを受け取る forward pass です。キャプチャ時と再生時の両方で呼ばれます。`path` パラメータは適切なエンコーダのサブモジュール（DeepSeek-OCR のグローバル / ローカルパスなど）へディスパッチします。
* `encoder_eager_forward(mm_kwargs, path="default")` — 適合するグラフがない場合の eager によるフォールバック forward です。`path` が `"global"` または `"local"` の場合、グラフキャプチャを行わずにそのエンコーダパスのみを実行します。
* `postprocess_encoder_output(..., local_output=None)` — エンコーダ出力を後処理します。`local_output` パラメータはローカルパスのエンコーダ出力テンソル（または `None`）を受け取り、dual-path モデルがグローバル特徴とローカル特徴を組み立てて画像ごとの最終的な埋め込みを作れるようにします。

!!! note
    `SupportsEncoderCudaGraph` プロトコルは、モデルに依存しないよう設計されています。新しい vision エンコーダのモデルは、マネージャを変更することなくプロトコルのメソッドを実装するだけで対応できます。

**サポートされるモデル:**

| アーキテクチャ | モデル | 画像の CG | 動画の CG | Dual-Path Graph |
| ------------ | ------ | ------------ | ------------ | --------------- |
| `DeepseekOCRForCausalLM` | `DeepSeek-OCR` | ✅︎ | ❌︎ | ✅︎ |
| `Gemma3ForConditionalGeneration` | `Gemma3` | ✅︎ | ❌︎ | ❌︎ |
| `Glm4vForConditionalGeneration` | `GLM-4.1V, GLM-4.6V-Flash` | ✅︎ | ✅︎ | ❌︎ |
| `InternVLChatModel` | `InternVL3.5`, `InternVL3`, `InternVL2.5`, `InternVL2` | ✅︎ | ✅︎ | ❌︎ |
| `KimiVLForConditionalGeneration` | `Kimi-VL` | ✅︎ | ❌︎ | ❌︎ |
| `Llama4ForConditionalGeneration` | `Llama 4` | ✅︎ | ❌︎ | ❌︎ |
| `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3_5ForConditionalGeneration` | `Qwen3.5`, `Qwen3.6` | ✅︎ | ✅︎ | ❌︎ |
| `Qwen3_5MoeForConditionalGeneration` | `Qwen3.5-MoE`, `Qwen3.6-MoE` | ✅︎ | ✅︎ | ❌︎ |
| `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ | ✅︎ |

!!! note
    エンコーダの CUDA Graphs は、現時点では Blackwell GPU 上で `--mm-encoder-attn-backend=FLASH_ATTN` および `--mm-encoder-attn-backend=FLASHINFER` を用いてテストされています。
    Qwen2-VL と Qwen2.5-VL については FA2 と FA3 のみテストされています。

## 設定 { #configuration }

`CompilationConfig` の次のフィールドがエンコーダの CUDA Graphs を制御します。

* `cudagraph_mm_encoder`（`bool`、既定値 `False`）— マルチモーダルエンコーダの CUDA Graph キャプチャを有効にします。有効にすると、各トークン budget レベルについてエンコーダの forward 全体を CUDA Graph としてキャプチャします。
* `encoder_cudagraph_token_budgets`（`list[int]`、既定値 `[]`）— キャプチャ対象のトークン budget レベルです。空（既定）の場合、モデルアーキテクチャから 2 のべき乗のレベルとして自動推定されます。ユーザーが指定した値は自動推定より優先されます。
* `encoder_cudagraph_max_vision_items_per_batch`（`int`、既定値 `0`）— キャプチャ時のバッチあたりの画像 / 動画の最大数です。0（既定）の場合、`max_budget // min_budget` として自動推定されます。
* `encoder_cudagraph_max_frames_per_batch`（`int`、既定値 `None`）— キャプチャ時のバッチあたりの動画フレームの最大数です。`None`（既定）の場合、`encoder_cudagraph_max_vision_items_per_batch * max_frames_per_video` として自動推定されます（`max_frames_per_video` は `EncoderCudaGraphConfig` に由来するモデル固有の値で、モデルの `get_max_frames_per_video()` によって計算されます）。プロンプトあたりの動画数を `0` に制限した場合、この値も `0` になります（つまり画像のみのモードにフォールバックします）。

dual-path モードは、モデル側の `EncoderCudaGraphConfig` のフィールド（`enable_dual_path_graph`、`global_token_per_image`、`local_token_per_patch`）で設定され、ユーザー側の追加設定は不要です。モデルが対応している場合、マネージャは自動的に別々の budget リストを生成し、dual-path の実行経路へ振り分けます。

## 使い方ガイド { #usage-guide }

### 画像の推論 { #image-inference }

`compilation_config` でエンコーダの CUDA Graphs を有効にします。

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

`Llama 4`（画像のみ）の場合:

```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --limit-mm-per-prompt '{"image": 1}' \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

budget を明示的に指定する場合:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

Python の例:

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # Optional: override auto-inferred budgets
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

マネージャはヒット / ミスの統計を追跡し、定期的にログ出力します。「ヒット」は画像が CUDA Graph の再生で処理されたことを、「ミス」は eager へのフォールバック（画像がすべての budget を超えた）を意味します。

### 動画の推論 { #video-inference }

`compilation_config` でエンコーダの CUDA Graphs を有効にします。

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true}'
```

budget を明示的に指定する場合:

```bash
vllm serve Qwen/Qwen3-VL-32B \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824], "encoder_cudagraph_max_vision_items_per_batch": 8, "encoder_cudagraph_max_frames_per_batch": 64}'
```

Python の例:

```python
import vllm

compilation_config = {
    "cudagraph_mm_encoder": True,
    # Optional: override auto-inferred budgets
    # "encoder_cudagraph_token_budgets": [2048, 4096, 8192, 13824],
    # "encoder_cudagraph_max_vision_items_per_batch": 8,
    # "encoder_cudagraph_max_frames_per_batch": 64,
}

model = vllm.LLM(
    model="Qwen/Qwen3-VL-32B",
    compilation_config=compilation_config,
)
```

## 性能について { #about-the-performance }

以下のベンチマークは、Blackwell GPU（GB200）上で `vllm bench mm-processor` を使って実行したものです。詳細は [#35963](https://github.com/vllm-project/vllm/pull/35963) を参照してください。

### 単一 GPU（GB200 × 1） { #single-gpu-1x-gb200 }

モデル: `Qwen/Qwen3-VL-30B-A3B-Instruct`、データセット: `lmarena-ai/VisionArena-Chat`（3000 プロンプト、ウォームアップ 300）、`max_model_len=32768`。

| バックエンド | 平均レイテンシの改善 | P99 レイテンシの改善 |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +11.8%（5.13→4.52ms） | +31.6%（9.16→6.26ms） |
| FLASHINFER | +19.6%（5.42→4.36ms） | +40.3%（10.87→6.49ms） |

再現方法:

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
  --num-prompts 3000 --num-warmups 300 \
  --max-model-len 32768 --seed 42 \
  --mm-encoder-attn-backend FLASH_ATTN \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

### 複数 GPU（GB200 × 4、TP=4、DP=4） { #multi-gpu-4x-gb200-tp4-dp4 }

モデル: `Qwen/Qwen3-VL-32B-Instruct`、データセット: `random-mm`（1000 プロンプト、ウォームアップ 200、リクエストあたり 336x336 の画像 20 枚）、`max_model_len=8192`。

| バックエンド | 平均レイテンシの改善 | P99 レイテンシの改善 |
| :------ | :----------------------- | :---------------------- |
| FLASH_ATTN | +18.4%（28.39→23.16ms） | +14.0%（238.78→205.28ms） |
| FLASHINFER | +44.4%（23.24→12.91ms） | +84.9%（172.41→26.05ms） |

再現方法:

```bash
vllm bench mm-processor \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --dataset-name random-mm \
  --random-mm-base-items-per-request 20 \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --random-mm-bucket-config '{"(336,336,1)": 1.0}' \
  --num-prompts 1000 --num-warmups 200 \
  --max-model-len 8192 --seed 42 \
  --mm-encoder-attn-backend FLASHINFER \
  --tensor-parallel-size 4 --mm-encoder-tp-mode data \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_token_budgets": [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4864], "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

!!! note
    動画推論に関する GPU（A100）でのベンチマークの詳細は [#38061](https://github.com/vllm-project/vllm/pull/38061) を参照してください。
