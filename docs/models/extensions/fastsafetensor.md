fastsafetensors によるモデル重みの読み込み
===================================================================

fastsafetensors ライブラリを使うと、GPU Direct Storage を活用してモデルの重みを GPU メモリへ直接読み込めます。詳細は[こちらの GitHub リポジトリ](https://github.com/foundation-model-stack/fastsafetensors)（英語）を参照してください。

この機能を有効にするには、コマンドライン引数 `--load-format fastsafetensors` を指定します。
