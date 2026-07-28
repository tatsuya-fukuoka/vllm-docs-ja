# InstantTensor によるモデル重みの読み込み { #loading-model-weights-with-instanttensor }

InstantTensor は、分散読み込み・パイプライン化されたプリフェッチ・ダイレクト I/O によって、CUDA デバイス上での Safetensors 形式の重みの読み込みを高速化します。利用可能な環境では GDS（GPUDirect Storage）にも対応します。
詳細は [InstantTensor の GitHub リポジトリ](https://github.com/scitix/InstantTensor)（英語）を参照してください。

## インストール { #installation }

```bash
pip install instanttensor
```

## vLLM で InstantTensor を使う { #use-instanttensor-in-vllm }

コマンドライン引数に `--load-format instanttensor` を追加します。

例:

```bash
vllm serve Qwen/Qwen2.5-0.5B --load-format instanttensor
```

## ベンチマーク { #benchmarks }

| モデル | GPU | バックエンド | 読み込み時間 (秒) | スループット (GB/s) | 高速化率 |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3-30B-A3B | 1*H200 | Safetensors | 57.4 | 1.1 | 1x |
| Qwen3-30B-A3B | 1*H200 | InstantTensor | 1.77 | 35 | <span style="color: green">**32.4x**</span> |
| DeepSeek-R1 | 8*H200 | Safetensors | 160 | 4.3 | 1x |
| DeepSeek-R1 | 8*H200 | InstantTensor | 15.3 | 45 | <span style="color: green">**10.5x**</span> |

完全なベンチマーク結果は <https://github.com/scitix/InstantTensor/blob/main/docs/benchmark.md>（英語）を参照してください。
