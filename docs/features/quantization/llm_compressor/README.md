# LLM Compressor { #llm-compressor }

[LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/) は、vLLM でのデプロイに向けてモデルを最適化するライブラリです。
FP4・FP8・INT8・INT4 など、幅広い量子化アルゴリズムをサポートしています。

## LLM Compressor を使う理由 { #why-use-llm-compressor }

現代の LLM は数十億のパラメータを 16 ビットや 32 ビットの浮動小数点で保持することが多く、大量の GPU メモリを必要とし、デプロイの選択肢を狭めます。
量子化はモデルの重みとアクティベーションの精度を小さなデータ型に落とすことで、推論の出力品質を保ちつつ必要なメモリを削減します。

LLM Compressor には次の利点があります。

- **メモリ使用量の削減**: より小さな GPU でより大きなモデルを動かせます。
- **推論コストの低減**: GPU あたりの同時ユーザー数が増え、本番環境でのクエリあたりのコストが直接下がります。
- **推論の高速化**: データ型が小さいほどメモリ帯域の消費が減り、特にメモリ律速のワークロードでスループットが向上します。

LLM Compressor は量子化・キャリブレーション・形式変換の複雑さを引き受け、vLLM ですぐに使えるモデルを生成します。

## 主な機能 { #key-features }

- **複数の量子化アルゴリズム**: AWQ、GPTQ、AutoRound、Round-to-Nearest をサポート。
QuIP や SpinQuant 形式の変換、KV キャッシュと Attention の量子化にも対応しています。
- **複数の量子化方式**: FP8、INT8、INT4、NVFP4、MXFP4、混合精度の量子化をサポート
- **ワンショット量子化**: 最小限のキャリブレーションデータで素早く量子化
- **vLLM との統合**: compressed-tensors 形式で、量子化したモデルを vLLM にシームレスにデプロイ
- **Hugging Face との互換性**: Hugging Face Hub のモデルで動作

## 参考資料 { #resources }

- [LLM Compressor の例](https://github.com/vllm-project/llm-compressor/tree/main/examples)（英語）
- [GitHub リポジトリ](https://github.com/vllm-project/llm-compressor)
