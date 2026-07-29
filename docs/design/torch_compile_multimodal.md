# マルチモーダルエンコーダにおける torch.compile { #torchcompile-with-multimodal-encoders }

`torch.compile` は、LLaMA 4 や Qwen-VL のような vision-language モデルをはじめとするエンコーダベースのアーキテクチャを含め、vLLM のマルチモーダルエンコーダやその他の nn モジュールにも適用できるようになりました。

このドキュメントでは、vLLM のマルチモーダルエンコーダに対する `torch.compile` 統合の基本的な仕組みと、性能向上のために新しいモデルへデコレータを適用する方法を説明します。

!!! note
    vLLM における `torch.compile` 統合の一般的な情報は、[torch.compile の設計ドキュメント](./torch_compile.md)を参照してください。

## 概要 { #overview }

最近、`@support_torch_compile` デコレータを、1 つのモデル種別の中の複数の nn モジュールコンポーネントに対して機能するようにしました。これにより、マルチモーダルエンコーダのコンパイルを有効にでき、スタックのより多くのコンポーネントで性能を改善できます。

[`Qwen2_5_vl`](https://github.com/vllm-project/vllm/pull/23207) の vision ブロックに適用したところ、コンパイル時間はいくらか増えるものの、エンドツーエンドで約 4.5% の性能向上が観測されました。

この機能は既定では無効ですが、`@support_torch_compile` デコレータが付いているモデルであれば、コンパイル設定で `compile_mm_encoder: true` を指定することで有効にできます。

## マルチモーダルコンポーネントにおけるコンパイルの仕組み { #how-compilation-works-for-multimodal-components }

### 有効化のための API { #apis-for-enablement }

エンコーダなどのマルチモーダルコンポーネントをコンパイルする場合、LLM のテキストバックボーンと同じ仕組みに、いくつかの足場を加えて使います。

1. `@support_torch_compile` デコレータに `enable_if=should_torch_compile_mm_encoder` を含めます。これにより、コンパイルが `compile_mm_encoder` 設定によって制御されます。

2. エンコーダのコンポーネントでは、`@support_torch_compile` デコレータに `is_encoder=True` を含めます。これはコンパイル範囲（compile range）の統合に必要です（「コンパイル範囲」を参照）。デコレータはクラス名をキャッシュディレクトリの接頭辞として自動的に使うため、独立してコンパイルされるサブモジュール同士（vision エンコーダのコンポーネントとテキストバックボーンなど）の衝突を防げます。

### CompilationConfig { #compilationconfig }

`compile_mm_encoder: true` を除き、マルチモーダルエンコーダはテキスト LLM と同じコンパイル設定を継承します。将来的には、より多くの設定に拡張する可能性があります。

## 新しいマルチモーダルモデル / コンポーネントへの torch.compile の適用 { #applying-torchcompile-to-a-new-multimodal-modelcomponent }

新しい一般的な nn.Module に `support_torch_compile` を適用する場合は、[`debug_vllm_compile`](./debug_vllm_compile.md) と同じ手順に従うことを推奨します。具体的には次のとおりです。

1. まず小さなモジュール（基本的な MLP 層など）に `support_torch_compile` を適用し、性能とのバランスが取れるところまで、より大きなモジュールへ範囲を広げていく

2. [`tlparse`](https://github.com/meta-pytorch/tlparse) を活用して、再コンパイルやグラフブレークの原因を特定して取り除く

3. `dynamic_arg_dims` と適切な `dynamic_shapes_config` を使って動的性を扱う

### よくある落とし穴 { #common-pitfalls }

## VllmBackend の機能サポート { #vllmbackend-feature-support }

### コンパイル範囲 { #compile-ranges }

torch.compile の統合では、動的形状に対するコンパイル範囲を max_batch_size から推測しようとします。しかしエンコーダで使われるモジュールでは、エンコーダが入力として受け取りうる形状の範囲が定まっていないため、この形状の推測が難しくなります。そこで `@support_torch_compile` デコレータの `is_encoder=True` によって、範囲を推測できないことを torch.compile に伝え、既定で (1, MAX_INT) の範囲を使います。

!!! note
    将来的には、性能向上のためにこの範囲をより狭めることを検討する可能性があります。

### CUDA graph { #cudagraphs }

CUDAGraph 統合を伴うマルチモーダルエンコーダのコンパイルはまだ検証しておらず、現時点で挙動は規定されていません。

## トラブルシューティング { #troubleshooting }

### vision エンコーダにおけるグラフブレーク { #graph-breaks-in-vision-encoders }

vision エンコーダの一部の演算はグラフブレークを引き起こすことがあります。特定するには次のようにします。

```bash
TORCH_LOGS="+dynamo" vllm serve <MODEL>
```

マルチモーダルモデルでグラフブレークが起きるよくある原因は次のとおりです。

- **画像サイズが動的**: 可変の解像度を扱うには `dynamic_shapes_config` を使ってください
- **トレースできない演算**: 一部の演算（to_list など）は Dynamo でサポートされていない場合があります
- **条件付きの処理**: 画像の属性に応じたデータ依存の分岐

### コンパイルエラー { #compilation-errors }

マルチモーダルモデルのコンパイルが失敗する場合は次のようにします。

1. **無効にして試す**: まず、コンパイルなしでモデルが動作することを確認します。
   ```bash
   vllm serve <model> --compilation-config='{"mode":0,"compile_mm_encoder":"false"}'
   ```

2. **ログを確認する**: デバッグログを有効にして、コンパイルの詳細を確認します。
   ```bash
   VLLM_LOGGING_LEVEL=DEBUG vllm serve <model> --compilation-config='{"compile_mm_encoder":"true"}'
   ```

3. **問題を報告する**: バグを見つけた場合は [GitHub で issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)してください。

## 関連項目 { #see-also }

- [torch.compile の統合](./torch_compile.md) - 中心となる設計ドキュメント
- [torch.compile のデバッグ](./debug_vllm_compile.md) - 詳細なデバッグガイド
- [マルチモーダル入力](../features/multimodal_inputs.md) - マルチモーダルデータの渡し方
- [エンコーダ分離](../features/disagg_encoder.md) - vision エンコーダのスケーリング
- [対応マルチモーダルモデル](../models/supported_models.md#list-of-multimodal-language-models) - モデルの互換性
