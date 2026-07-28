# vLLM-Project/Speculators { #vllm-projectspeculators }

![User Flow Light](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/speculative_decoding/speculators-user-flow-light.svg#only-light)
![User Flow Dark](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/speculative_decoding/speculators-user-flow-dark.svg#only-dark)

[Speculators](https://docs.vllm.ai/projects/speculators/en/latest/) は、投機的デコーディングによって LLM 推論を高速化するライブラリです。効率的なドラフトモデルの学習を提供し、vLLM とシームレスに統合してレイテンシの低減とスループットの向上を実現します。

Speculators の主な機能は次のとおりです。

- **vLLM を使ったオフラインの学習データ生成**: vLLM で隠れ状態を生成できます。データのサンプルはディスクに保存され、ドラフトモデルの学習に使えます。
- **ドラフトモデルの学習サポート**: 単層・多層のドラフトモデルをエンドツーエンドで学習できます。MoE でないモデルと MoE モデルの両方に対応しています。
- **標準化された拡張可能な形式**: 投機用モデルを定義するための Hugging Face 互換の形式を提供します。外部の研究リポジトリの形式を標準の speculators 形式へ変換するツールもあり、導入が容易です。
- **vLLM とのシームレスな統合**: vLLM へ直接デプロイできるよう設計されており、最小限のオーバーヘッドで低レイテンシかつ本番品質の推論を実現します。

## Speculators を使う理由 { #why-use-speculators }

大規模言語モデルはトークンを 1 つずつ生成するため、本質的なボトルネックが生じます。各トークンでモデル全体の forward が必要になり、メモリ律速の処理を待つ間 GPU の計算資源が余ってしまいます。
投機的デコーディングは、より小さく高速な「ドラフト」モデル（多くの場合、Transformer の 1 層だけ）で先のトークンをまとめて予測し、本体のモデルで並列に検証することでこれに対処します。

投機的デコーディングには次の利点があります。

- **レイテンシの低減**: チャットボットやコードアシスタントのように応答時間がそのまま体験に響く対話的なアプリケーションで、トークン生成が 2〜3 倍速くなります
- **GPU 使用率の向上**: 大きなモデルにおけるレイテンシ律速・メモリ律速の Decode を、計算律速の並列トークン検証に置き換え、ハードウェアの使用効率を高めます。
- **品質の劣化なし**: 投機的デコーディングはターゲットモデルを近似しません。受理されたトークンは、同じサンプリング設定でターゲットモデルが生成したはずのトークンそのものです。受理されなかったドラフトトークンは破棄され、ターゲットモデルが生成し直します。
- **コスト効率**: 各リクエストがハードウェアを占有する時間が短くなり、GPU あたりで処理できるリクエスト数が増えます

Speculators は、対話 AI、対話的なコーディングアシスタント、ストリーミングのテキスト生成など、ユーザーがリアルタイムに応答を待つレイテンシ重視のアプリケーションで特に有効です。

## 参考資料 { #resources }

- [Speculators の例](https://github.com/vllm-project/speculators/tree/main/examples)（英語）
- [GitHub リポジトリ](https://github.com/vllm-project/speculators)
