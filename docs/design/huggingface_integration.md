# Hugging Face との統合 { #integration-with-hugging-face }

このドキュメントでは、vLLM が Hugging Face のライブラリとどのように統合されているかを説明します。`vllm serve` を実行したときに内部で何が起きるのかを、順を追って見ていきます。

たとえば、`vllm serve Qwen/Qwen2-7B` を実行して人気のある Qwen モデルをサービングするとします。

1. `model` 引数は `Qwen/Qwen2-7B` です。vLLM は、対応する設定ファイル `config.json` の有無を調べて、このモデルが存在するかどうかを判定します。実装は[このコードスニペット](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L162-L182)を参照してください。この処理では次のことが行われます。
    - `model` 引数が既存のローカルパスに対応する場合、vLLM はそのパスから設定ファイルを直接読み込みます。
    - `model` 引数がユーザー名とモデル名からなる Hugging Face のモデル ID である場合、vLLM はまず Hugging Face のローカルキャッシュにある設定ファイルを使おうとします。このとき `model` 引数をモデル名、`--revision` 引数をリビジョンとして扱います。Hugging Face のキャッシュの仕組みについては[公式サイト](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome)を参照してください。
    - `model` 引数が Hugging Face のモデル ID で、キャッシュに見つからない場合、vLLM は Hugging Face のモデルハブから設定ファイルをダウンロードします。実装は[この関数](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L91)を参照してください。入力引数には、モデル名としての `model` 引数、リビジョンとしての `--revision` 引数、そしてモデルハブへのアクセストークンとしての環境変数 `HF_TOKEN` が含まれます。今回の例では、vLLM は [config.json](https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json) をダウンロードします。

2. モデルの存在を確認したあと、vLLM はその設定ファイルを読み込み、辞書に変換します。実装は[このコードスニペット](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L185-L186)を参照してください。

3. 次に、vLLM は設定辞書の `model_type` フィールドを[調べ](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L189)、使用する設定オブジェクトを[生成](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L190-L216)します。vLLM が直接サポートしている `model_type` の値がいくつかあり、その一覧は[こちら](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/transformers_utils/config.py#L48)にあります。`model_type` が一覧にない場合、vLLM は [AutoConfig.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained) を使い、`model`、`--revision`、`--trust_remote_code` を引数として設定クラスを読み込みます。次の点に注意してください。
    - Hugging Face 側にも、使用する設定クラスを決めるための独自のロジックがあります。ここでも `model_type` フィールドを使って transformers ライブラリ内のクラス名を検索します。サポートされているモデルの一覧は[こちら](https://github.com/huggingface/transformers/tree/main/src/transformers/models)です。`model_type` が見つからない場合、Hugging Face は設定 JSON ファイルの `auto_map` フィールドを使ってクラス名を決定します。具体的には `auto_map` の下の `AutoConfig` フィールドです。例として [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2.5/blob/main/config.json) を参照してください。
    - `auto_map` の下の `AutoConfig` フィールドは、モデルのリポジトリ内のモジュールパスを指します。設定クラスを作るために、Hugging Face はそのモジュールを import し、`from_pretrained` メソッドで設定クラスを読み込みます。これは一般に任意コードの実行につながりうるため、`--trust_remote_code` が有効な場合にのみ実行されます。

4. 続いて、vLLM は設定オブジェクトに対していくつかの歴史的なパッチを適用します。その多くは RoPE の設定に関するものです。実装は[こちら](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/config.py#L244)を参照してください。

5. 最後に、vLLM は初期化すべきモデルクラスにたどり着けます。vLLM は設定オブジェクトの `architectures` フィールドを使って初期化するモデルクラスを決定します。アーキテクチャ名からモデルクラスへの対応は[レジストリ](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/registry.py#L80)で管理されています。アーキテクチャ名がレジストリに見つからない場合、そのモデルアーキテクチャは vLLM でサポートされていないということです。`Qwen/Qwen2-7B` の場合、`architectures` フィールドは `["Qwen2ForCausalLM"]` であり、[vLLM のコード](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/model_executor/models/qwen2.py#L364)の `Qwen2ForCausalLM` クラスに対応します。このクラスは、さまざまな設定に応じて自身を初期化します。

これ以外にも、vLLM が Hugging Face に依存しているものが 2 つあります。

1. **トークナイザー**: vLLM は入力テキストのトークン化に Hugging Face のトークナイザーを使います。トークナイザーは [AutoTokenizer.from_pretrained](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained) を使い、`model` 引数をモデル名、`--revision` 引数をリビジョンとして読み込まれます。`vllm serve` コマンドで `--tokenizer` 引数を指定すれば、別のモデルのトークナイザーを使うこともできます。関連する引数として `--tokenizer-revision` と `--tokenizer-mode` があります。`VLLM_USE_FASTOKENS=1` を設定すると、vLLM が読み込む HF の fast トークナイザーが、そのまま置き換え可能な Rust 製 BPE バックエンドに切り替わります（[fastokens バックエンド](../configuration/optimization.md#fastokens-backend)を参照）。これらの引数の意味は Hugging Face のドキュメントを確認してください。このロジックの該当箇所は [get_tokenizer](https://github.com/vllm-project/vllm/blob/127c07480ecea15e4c2990820c457807ff78a057/vllm/transformers_utils/tokenizer.py#L87) 関数にあります。トークナイザーを取得したあと、vLLM はトークナイザーの計算コストが高い属性の一部を [vllm.tokenizers.hf.get_cached_tokenizer][] でキャッシュします。

2. **モデルの重み**: vLLM は Hugging Face のモデルハブから、`model` 引数をモデル名、`--revision` 引数をリビジョンとしてモデルの重みをダウンロードします。モデルハブからどのファイルをダウンロードするかは `--load-format` 引数で制御できます。既定では safetensors 形式の重みを読み込もうとし、safetensors 形式が利用できない場合は PyTorch の bin 形式にフォールバックします。`--load-format dummy` を渡せば重みのダウンロードをスキップできます。
    - safetensors 形式の利用を推奨します。分散推論での読み込みが効率的で、任意コードの実行に対しても安全だからです。safetensors 形式の詳細は[ドキュメント](https://huggingface.co/docs/safetensors/en/index)を参照してください。このロジックの該当箇所は[こちら](https://github.com/vllm-project/vllm/blob/10b67d865d92e376956345becafc249d4c3c0ab7/vllm/model_executor/model_loader/loader.py#L385)です。次の点に注意してください。

以上が、vLLM と Hugging Face の統合のすべてです。

まとめると、vLLM は設定ファイル `config.json`、トークナイザー、モデルの重みを Hugging Face のモデルハブまたはローカルディレクトリから読み込みます。設定クラスは vLLM のもの、Hugging Face transformers のもの、あるいはモデルのリポジトリから読み込んだものが使われます。
