# FP8 ViT エンコーダ Attention { #fp8-vit-encoder-attention }

大きな画像（QHD、4K など）を扱い、テキストのプロンプトや生成が比較的短い視覚理解のワークロードでは、ViT エンコーダの Attention が大きなボトルネックになることがあります。特にテキストモデルが量子化されている（NVFP4 など）場合に顕著です。vLLM は FlashInfer の cuDNN バックエンドを通じて、ViT エンコーダ Attention の FP8 量子化をオプションとしてサポートしています。Q/K/V は cuDNN の Attention 呼び出しの直前にその場で FP8 に量子化されます。

!!! note
    - 現時点では Qwen3-VL 系のモデルのみ対応しています（`qwen3_vl`、`qwen3_vl_moe`、
      `qwen3_5`、`qwen3_5_moe`、および Qwen3 ViT を使う他のモデル）。
    - 動的スケーリングは ViT の full CUDA graph とは併用できません。
    - 性能向上が見られるのは主に QHD / 4K の解像度、または複数画像を含むリクエストの場合です。
      小さい画像では量子化のオーバーヘッド（量子化カーネルの起動 3 回 + アンパディング）により
      高速化が得られないことがあります。
    - FP8 の Tensor Core による高速化は、GB200 よりも GB300 でより顕著です。

## 要件 { #requirements }

- FlashInfer の cuDNN バックエンド（cuDNN 9.17.1 以上）。

## 使い方 { #usage }

`--mm-encoder-attn-backend FLASHINFER` と併せて `--mm-encoder-attn-dtype fp8` を指定すると、FP8 の ViT Attention が有効になります。

```bash
vllm serve $MODEL \
    --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8
```

既定（スケールファイルなし）では **動的スケーリング** が使われます。観測された Q/K/V の amax 値を保持する 16 エントリのリングバッファをもとに、forward ごとにスケールが更新されます。キャリブレーションなしで BF16 と同等の精度が得られますが、forward ごとにわずかなオーバーヘッドが加わります。

## 1 回キャリブレーションして再利用するワークフロー（推奨） { #calibrate-once-reuse-workflow-recommended }

本番環境では、代表的なデータセットで静的スケールを一度キャリブレーションしておき、それを再利用することで動的スケーリングのオーバーヘッドを避けられます。

```bash
# Step 1: calibrate and save scales (runs dynamic scaling for 16 passes,
# then dumps the learned scales to JSON).
vllm bench mm-processor \
    --model $MODEL --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8 \
    --mm-encoder-fp8-scale-save-path /path/to/scales.json \
    --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
    --num-prompts 100

# Step 2: serve with static scales (no dynamic overhead).
vllm serve $MODEL \
    --mm-encoder-attn-backend FLASHINFER \
    --mm-encoder-attn-dtype fp8 \
    --mm-encoder-fp8-scale-path /path/to/scales.json
```

保存されるスケールには `--mm-encoder-fp8-scale-save-margin`（既定値 `1.5`）が乗算され、キャリブレーションセットに現れない活性値の外れ値に対する余裕を持たせます。この既定値はデータセットをまたいでも汎化することが検証されています（たとえば VisionArena-Chat でキャリブレーションしたスケールは、ChartQA でも BF16 と同等の精度を保ちます）。

## スケールファイルの形式 { #scale-file-format }

```json
{
    "visual.blocks.0.attn.attn": {"q": 224.0, "k": 198.0, "v": 210.0},
    "visual.blocks.1.attn.attn": {"q": 218.0, "k": 195.0, "v": 207.0}
}
```

キーとして `q_scale` / `k_scale` / `v_scale` もエイリアスとして受け付けられます。

## 性能 { #performance }

**cuDNN の Attention カーネル単体**（PyTorch profiler、`cudnn_generated_fort_native_sdpa_sm100_flash_fprop`、head_dim=128、seq_len=8192）:

| ハードウェア | BF16 | FP8 | 高速化 |
| -------- | ---- | ---- | ------- |
| GB200 | 350 us | 312 us | **1.12x** |
| GB300 | 300 us | 211 us | **1.42x** |

**エンコーダ forward のエンドツーエンド時間**（GB200 上の Qwen3-VL-30B-A3B-Instruct、リクエストあたり 3 画像）:

| 解像度 | BF16 中央値 | FP8 中央値 | 高速化 |
| ---------- | ----------- | ---------- | ------- |
| HD (720x1280) | 31.77 ms | 36.39 ms | 0.87x |
| FullHD (1080x1920) | 57.99 ms | 58.73 ms | ~same |
| QHD (1440x2560) | 131.83 ms | 122.30 ms | **1.08x** |
| 4K (2160x3840) | 543.44 ms | 460.31 ms | **1.18x** |

リクエストあたり 3 画像の場合、損益分岐点は FullHD 付近です。QHD 以上では FP8 が有利になります。

## 精度 { #accuracy }

ChartQA、Qwen3-VL-8B-Instruct、500 サンプルでの結果です。FP8 static は VisionArena-Chat でキャリブレーションしたスケール（既定の 1.5 倍マージン）を使っています。

| 指標 | BF16 | FP8 動的 | FP8 静的 |
| ------ | ---- | ----------- | ---------- |
| relaxed_accuracy | 0.780 | 0.776 | 0.780 |
| anywhere_accuracy | 0.806 | 0.816 | 0.814 |
| exact_match | 0.584 | 0.582 | 0.578 |

3 つの構成はいずれも統計的な誤差の範囲で一致しており、あるデータセットでキャリブレーションした静的スケールが別のデータセットにも汎化することが確認できます。
