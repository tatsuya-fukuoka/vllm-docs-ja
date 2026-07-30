# マルチモーダルデータの処理 { #multi-modal-data-processing }

[チャンク化プレフィル](../configuration/optimization.md#chunked-prefill)や[プレフィックスキャッシュ](../features/automatic_prefix_caching.md)といった vLLM のさまざまな最適化を可能にするため、[`BaseMultiModalProcessor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor) を使い、HF プロセッサの出力にもとづいて、プレースホルダーの特徴トークン（`<image>` など）とマルチモーダル入力（生の入力画像など）との対応関係を提供します。

[`BaseMultiModalProcessor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor) の主な機能は次のとおりです。

## プロンプト更新の検出 { #prompt-update-detection }

HF プロセッサの主な役割の 1 つは、プレースホルダートークンでプロンプトを更新することです。たとえば次のような処理です。

- 文字列の先頭に特徴プレースホルダートークン（`<image><image>...<image>` など。個数は特徴サイズと等しい）を挿入する。
- 既存の入力プレースホルダートークン（画像 1 枚に対する `<image>` など）を、特徴プレースホルダートークン（`<image><image>...<image>` など。個数は特徴サイズと等しい）に置き換える。

どのトークンが更新されたかという情報は、プレースホルダーの特徴トークンとマルチモーダル入力の対応関係を見つけるうえで鍵になります。

vLLM では、この情報を [`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) 内の [`PromptUpdate`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptUpdate) で指定します。更新後のトークンが存在するかどうかを確認することで、HF がプロンプトを更新したかどうかを自動的に検出できます。

## トークン化済みプロンプトの入力 { #tokenized-prompt-inputs }

トークン化を別プロセスで行えるようにするため、マルチモーダルデータと併せて入力トークン ID を渡すこともサポートしています。

### 課題 { #the-problem }

HF プロセッサは主に次の手順をたどります。

1. テキストをトークン化する
2. マルチモーダル入力を処理する
3. プロンプトの更新を行う

そして、次のことが求められます。

- テキスト + マルチモーダル入力の場合は、手順 1〜3 をすべて適用する。
- トークン化済み + マルチモーダル入力の場合は、手順 2〜3 のみを適用する。

HF プロセッサを書き換えずに、これをどう実現すればよいでしょうか。異なる入力に対して HF プロセッサを複数回呼び出す方法が考えられます。

- テキスト + マルチモーダル入力の場合は、HF プロセッサをそのまま呼び出す。
- トークン化済み + マルチモーダル入力の場合は、マルチモーダル入力に対してのみプロセッサを呼び出す。

HF プロセッサはテキスト + マルチモーダル入力をネイティブにサポートしていますが、トークン化済み + マルチモーダル入力についてはそうではありません。入力プレースホルダートークンの数がマルチモーダル入力の数と一致しないとエラーになります。

さらに、トークン化済みのテキストは HF プロセッサを通っていないため、出力トークンとマルチモーダルデータの整合性を保つには、手順 3 を自分たちで適用する必要があります。

### ダミーテキスト { #dummy-text }

1 つ目の課題については、各モデルに対して、マルチモーダル入力の数にもとづくダミーテキストの生成方法を [`get_dummy_text`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text) で定義することを求めることで回避しています。これにより、マルチモーダル入力に対応するダミーテキストを生成し、両者をまとめて入力して処理済みのマルチモーダルデータを得られます。

### プロンプトの自動更新 { #automatic-prompt-updating }

2 つ目の課題には、[`_apply_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates) にモデル非依存のコードを実装することで対処しています。ここでは、[`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) が出力する仕様にもとづいて、特徴プレースホルダートークンでプロンプトを自動的に更新します。

### まとめ { #summary }

ダミーテキストとプロンプトの自動更新のおかげで、vLLM のマルチモーダルプロセッサは、マルチモーダルデータを伴うテキストプロンプトとトークンプロンプトの両方を受け付けられるようになりました。詳細なロジックは [`_apply_hf_processor_main`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main) に示されています。

## プロセッサ出力のキャッシュ { #processor-output-caching }

Qwen2-VL 向けのものなど、一部の HF プロセッサは[非常に低速](https://github.com/vllm-project/vllm/issues/9238)です。この問題を緩和するため、HF プロセッサのマルチモーダル出力をキャッシュし、同じマルチモーダル入力（画像など）を再度処理しないようにしています。

新しいデータが渡されると、まずどの項目がキャッシュにあり、どの項目が欠けているかを確認します。欠けている項目は 1 つのバッチとして HF プロセッサに渡されてキャッシュされ、その後キャッシュ内の既存項目とマージされます。

欠けているマルチモーダルデータ項目だけを処理するため、入力プレースホルダートークンの数はマルチモーダル入力の数と一致しなくなり、テキストプロンプトと一緒に HF プロセッサへ渡すことができません。そこで、テキストとマルチモーダル入力を別々に処理し、HF のエラーを避けるために[ダミーテキスト](#dummy-text)を使います。この方法では HF のプロンプト更新処理が飛ばされるため、あとから[プロンプトの自動更新](#automatic-prompt-updating)を適用し、出力トークンとマルチモーダルデータの整合性を保ちます。
