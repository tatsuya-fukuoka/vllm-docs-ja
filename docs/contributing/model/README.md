# 概要 { #summary }

!!! important
    多くのデコーダー型の言語モデルは、vLLM に実装しなくても [Transformers モデリングバックエンド](../../models/supported_models.md#transformers)で自動的に読み込めるようになりました。まずは `vllm serve <model>` が動くか試してみてください。

vLLM のモデルは、性能を最適化するためのさまざまな[機能](../../features/README.md#compatibility-matrix)を活用する、特殊化された [PyTorch](https://pytorch.org/) のモデルです。

モデルを vLLM に統合する難しさは、そのモデルのアーキテクチャに大きく依存します。
vLLM の既存モデルと似たアーキテクチャであれば、作業はかなり簡単です。
一方、新しい演算子（新しい Attention の機構など）を含むモデルでは複雑になります。

手順ごとのガイドは次のページを参照してください。

- [基本的なモデル](basic.md)
- [モデルの登録](registration.md)
- [ユニットテスト](tests.md)
- [マルチモーダル対応](multimodal.md)
- [音声認識対応](transcription.md)

!!! tip
    モデルの統合で問題が起きた場合は、遠慮なく [GitHub の Issue](https://github.com/vllm-project/vllm/issues) を作成するか、
    [開発者向け Slack](https://slack.vllm.ai) で質問してください。
    喜んでお手伝いします。
