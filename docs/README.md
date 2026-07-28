---
hide:
  - navigation
  - toc
---

# vLLM へようこそ { #welcome-to-vllm }

<figure markdown="span">
  ![](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/logos/vllm-logo-text-light.png){ align="center" alt="vLLM Light" class="logo-light" width="60%" }
  ![](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/logos/vllm-logo-text-dark.png){ align="center" alt="vLLM Dark" class="logo-dark" width="60%" }
</figure>

<p style="text-align:center">
<strong>誰にとっても簡単・高速・低コストな LLM サービング
</strong>
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

vLLM は、LLM の推論とサービングのための高速で使いやすいライブラリです。

vLLM はカリフォルニア大学バークレー校の [Sky Computing Lab](https://sky.cs.berkeley.edu) で開発が始まり、現在では数十の大学・企業と 2000 名を超えるコントリビューターからなる多様なコミュニティによって開発・保守される、もっとも活発なオープンソース AI プロジェクトのひとつに成長しました。

vLLM をどこから始めるとよいかは、目的によって異なります。

- vLLM でオープンソースのモデルを動かしたい場合は、[クイックスタート](./getting_started/quickstart.md)から始めることをおすすめします
- vLLM を使ってアプリケーションを作りたい場合は、[ユーザーガイド](./usage/README.md)から始めることをおすすめします
- vLLM 自体を開発したい場合は、[開発者ガイド](./contributing/README.md)から始めることをおすすめします

vLLM の開発状況については以下を参照してください。

- [ロードマップ](https://roadmap.vllm.ai)
- [リリース](https://github.com/vllm-project/vllm/releases)

vLLM が高速である理由:

- 最高水準のサービングスループット
- [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html) による Attention の key / value メモリの効率的な管理
- 受信リクエストの連続バッチング、チャンク化 Prefill、プレフィックスキャッシュ
- piecewise / full CUDA・HIP グラフによる高速かつ柔軟なモデル実行
- 量子化: FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、compressed-tensors、ModelOpt、TorchAO ほか[多数](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA、Triton などの最適化された Attention カーネル
- CUTLASS、TRTLLM-GEN、CuTeDSL を用いた、各種精度向けの最適化された GEMM / MoE カーネル
- n-gram、suffix、EAGLE、DFlash などの投機的デコーディング
- torch.compile によるカーネルの自動生成とグラフレベルの変換
- Prefill・Decode・Encode の分離実行

vLLM が柔軟で使いやすい理由:

- 主要な Hugging Face モデルとのシームレスな連携
- *パラレルサンプリング*、*ビームサーチ*などさまざまなデコーディングアルゴリズムによる高スループットなサービング
- 分散推論のためのテンソル並列・パイプライン並列・データ並列・エキスパート並列・コンテキスト並列
- ストリーミング出力
- xgrammar または guidance を用いた構造化出力の生成
- ツール呼び出しと reasoning パーサー
- OpenAI 互換 API サーバー、加えて Anthropic Messages API と gRPC のサポート
- Dense 層・MoE 層に対する効率的なマルチ LoRA サポート
- NVIDIA GPU、AMD GPU、x86/ARM/PowerPC CPU のサポート。さらに Google TPU、Intel Gaudi、IBM Spyre、Huawei Ascend、Rebellions NPU、Apple Silicon、MetaX GPU など多様なハードウェアプラグインにも対応

vLLM は HuggingFace 上の 200 を超えるモデルアーキテクチャをシームレスにサポートしています。例:

- デコーダーのみの LLM（Llama、Qwen、Gemma など）
- Mixture-of-Expert LLM（Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS など）
- Attention と状態空間モデルのハイブリッド（Mamba、Qwen3.5 など）
- マルチモーダルモデル（LLaVA、Qwen-VL、Pixtral など）
- 埋め込み・検索モデル（E5-Mistral、GTE、ColBERT など）
- 報酬モデル・分類モデル（Qwen-Math など）

対応モデルの一覧は[こちら](./models/supported_models.md)を参照してください。

さらに詳しい情報:

- [vLLM 公開時のブログ記事](https://blog.vllm.ai/2023/06/20/vllm.html)（PagedAttention の紹介）
- [vLLM の論文](https://arxiv.org/abs/2309.06180)（SOSP 2023）
- [How continuous batching enables 23x throughput in LLM inference while reducing p50 latency](https://www.anyscale.com/blog/continuous-batching-llm-inference)（Cade Daniel 他）
- [vLLM Meetups](community/meetups.md)
