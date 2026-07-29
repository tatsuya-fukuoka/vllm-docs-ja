# LoRA アダプタ { #lora-adapters }

このドキュメントでは、ベースモデルの上で [LoRA アダプタ](https://arxiv.org/abs/2106.09685)を vLLM とともに使う方法を説明します。

LoRA アダプタは、[`SupportsLoRA`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsLoRA) を実装している任意の vLLM モデルで利用できます。

アダプタは、わずかなオーバーヘッドでリクエストごとに効率よくサービングできます。まず、アダプタをダウンロードしてローカルに保存します。

```python
from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")
```

次に、ベースモデルをインスタンス化し、`enable_lora=True` フラグを渡します。

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)
```

あとはプロンプトを投入し、`lora_request` パラメータを付けて `llm.generate` を呼び出します。`LoRARequest` の第 1 引数は人が識別できる名前、第 2 引数はアダプタのグローバルに一意な ID、第 3 引数は LoRA アダプタへのパスです。

??? code

    ```python
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"],
    )

    prompts = [
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    )
    ```

非同期エンジンで LoRA アダプタを使う方法や、より高度な設定オプションの使い方の例は [examples/features/lora/multilora_offline.py](../../examples/features/lora/multilora_offline.py) を参照してください。

## LoRA アダプタのサービング { #serving-lora-adapters }

LoRA を適用したモデルは、OpenAI 互換の vLLM サーバーでもサービングできます。そのためには、サーバー起動時に `--lora-modules {name}={path} {name}={path}` で各 LoRA モジュールを指定します。

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

サーバーのエントリポイントは、その他の LoRA 設定パラメータ（`max_loras`、`max_lora_rank`、`max_cpu_loras` など）も受け付け、これらは以降のすべてのリクエストに適用されます。`/models` エンドポイントに問い合わせると、LoRA とそのベースモデルが表示されるはずです（`jq` が未インストールの場合は、[このガイド](https://jqlang.org/download/)に従ってインストールできます）。

??? console "コマンド"

    ```bash
    curl localhost:8000/v1/models | jq .
    {
        "object": "list",
        "data": [
            {
                "id": "meta-llama/Llama-3.2-3B-Instruct",
                "object": "model",
                ...
            },
            {
                "id": "sql-lora",
                "object": "model",
                ...
            }
        ]
    }
    ```

リクエストでは、`model` リクエストパラメータを通じて、他のモデルと同じように LoRA アダプタを指定できます。リクエストはサーバー全体の LoRA 設定に従って処理されます（つまり、ベースモデルへのリクエストと並行して、さらに `max_loras` が十分に大きく設定されていれば他の LoRA アダプタへのリクエストとも並行して処理されます）。

リクエストの例は次のとおりです。

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq
```

## LoRA アダプタの動的なサービング { #dynamically-serving-lora-adapters }

vLLM サーバーは、サーバー起動時に LoRA アダプタをサービングするだけでなく、専用の API エンドポイントやプラグインを通じて実行時に LoRA アダプタを動的に構成することもできます。この機能は、モデルをその場で切り替える柔軟性が必要な場合にとくに役立ちます。

!!! warning
    この機能にはセキュリティ上のリスクがあります。隔離された、完全に信頼できる環境でない限り、本番では使用しないでください。

動的な LoRA 設定を有効にするには、環境変数 `VLLM_ALLOW_RUNTIME_LORA_UPDATING` が `True` に設定されていることを確認してください。

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### API エンドポイントを使う { #using-api-endpoints }

LoRA アダプタの読み込み:

LoRA アダプタを動的に読み込むには、読み込むアダプタの情報を添えて `/v1/load_lora_adapter` エンドポイントへ POST リクエストを送ります。リクエストのペイロードには、LoRA アダプタの名前とパスを含めます。

LoRA アダプタを読み込むリクエストの例:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
}'
```

リクエストが成功すると、`vllm serve` は `200 OK` のステータスコードを返し、`curl` にはレスポンスボディ `Success: LoRA adapter 'sql_adapter' added successfully` が表示されます。アダプタが見つからない、読み込めないなどのエラーが起きた場合は、適切なエラーメッセージが返されます。

LoRA アダプタの取り外し:

すでに読み込まれている LoRA アダプタを取り外すには、取り外すアダプタの名前または ID を添えて `/v1/unload_lora_adapter` エンドポイントへ POST リクエストを送ります。

リクエストが成功すると、`vllm serve` は `200 OK` のステータスコードを返し、`curl` にはレスポンスボディ `Success: LoRA adapter 'sql_adapter' removed successfully` が表示されます。

LoRA アダプタを取り外すリクエストの例:

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "sql_adapter"
}'
```

### プラグインを使う { #using-plugins }

代わりに、LoRAResolver プラグインを使って LoRA アダプタを動的に読み込むこともできます。LoRAResolver プラグインを使うと、ローカルファイルシステムや S3 など、ローカル・リモート双方のソースから LoRA アダプタを読み込めます。リクエストのたびに、まだ読み込まれていない新しいモデル名が現れると、LoRAResolver は対応する LoRA アダプタの解決と読み込みを試みます。

異なるソースから LoRA アダプタを読み込みたい場合は、複数の LoRAResolver プラグインを設定できます。たとえば、ローカルファイル用のリゾルバと S3 ストレージ用のリゾルバを用意する、といった具合です。vLLM は最初に見つかった LoRA アダプタを読み込みます。

既存のプラグインをインストールすることも、自作することもできます。vLLM には既定で、[ローカルディレクトリから LoRA アダプタを読み込むリゾルバプラグインと、Hugging Face Hub のリポジトリから LoRA アダプタを読み込むリゾルバプラグイン](https://github.com/vllm-project/vllm/tree/main/vllm/plugins/lora_resolvers)が同梱されています。
いずれのリゾルバを使う場合も、`VLLM_ALLOW_RUNTIME_LORA_UPDATING` を True に設定する必要があります。

- ローカルディレクトリを使うには、`VLLM_PLUGINS` に `lora_filesystem_resolver` を含め、`VLLM_LORA_RESOLVER_CACHE_DIR` にローカルディレクトリを設定します。vLLM が LoRA アダプタ `foobar` を使うリクエストを受け取ると、まずローカルディレクトリ内に `foobar` というディレクトリがないか探し、その中身を LoRA アダプタとして読み込もうとします。成功すればリクエストは通常どおり完了し、以降そのアダプタはサーバー上で通常どおり利用できるようになります。
- Hugging Face Hub のリポジトリを使うには、`VLLM_PLUGINS` に `lora_hf_hub_resolver` を含め、`VLLM_LORA_RESOLVER_HF_REPO_LIST` に Hugging Face Hub のリポジトリ ID をカンマ区切りで設定します。vLLM が LoRA アダプタ `my/repo/subpath` へのリクエストを受け取ると、`my/repo` の `subpath` にアダプタが存在し `adapter_config.json` を含んでいればそれをダウンロードし、`lora_filesystem_resolver` と同様にキャッシュディレクトリに対するリクエストを構築します。なお、リモートからのダウンロードを有効にすることは安全ではなく、本番環境での利用は想定されていません。

あるいは、次の手順例に従って自分でプラグインを実装することもできます。

1. LoRAResolver インターフェースを実装します。

    ??? code "シンプルな S3 向け LoRAResolver 実装の例"

        ```python
        import os
        import s3fs
        from vllm.lora.request import LoRARequest
        from vllm.lora.resolver import LoRAResolver

        class S3LoRAResolver(LoRAResolver):
            def __init__(self):
                self.s3 = s3fs.S3FileSystem()
                self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
                self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

            async def resolve_lora(self, base_model_name, lora_name):
                s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
                local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)

                # Download the LoRA from S3 to the local path
                await self.s3._get(
                    s3_path, local_path, recursive=True, maxdepth=1
                )

                lora_request = LoRARequest(
                    lora_name=lora_name,
                    lora_path=local_path,
                    lora_int_id=abs(hash(lora_name)),
                )
                return lora_request
        ```

2. `LoRAResolver` プラグインを登録します。

    ```python
    from vllm.lora.resolver import LoRAResolverRegistry

    s3_resolver = S3LoRAResolver()
    LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
    ```

    詳細は [vLLM のプラグインシステム](../design/plugin_system.md)を参照してください。

### LoRA のインプレース再読み込み { #in-place-lora-reloading }

LoRA アダプタを動的に読み込む際、同じ名前を保ったまま既存のアダプタを更新後の重みで置き換えたい場合があります。`load_inplace` パラメータがこれを可能にします。これは、推論を中断せずにアダプタを継続的に更新・差し替えする非同期の強化学習の構成でよく必要になります。

`load_inplace=True` の場合、vLLM は既存のアダプタを新しいものに置き換えます。

同じ名前で LoRA アダプタを読み込む / 置き換えるリクエストの例:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "my-adapter",
    "lora_path": "/path/to/adapter/v2",
    "load_inplace": true
}'
```

## `--lora-modules` の新しい形式 { #new-format-for-lora-modules }

以前のバージョンでは、LoRA モジュールをキーと値のペア、または JSON 形式で次のように指定していました。例:

```bash
--lora-modules  sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

この形式では各 LoRA モジュールの `name` と `path` しか指定できず、`base_model_name` を指定する方法がありませんでした。
現在は、JSON 形式で name と path に加えて base_model_name を指定できます。例:

```bash
--lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

後方互換性のため、従来のキーと値の形式（name=path）も引き続き使えますが、その場合 `base_model_name` は未指定のままになります。

## 2D と 3D の MoE LoRA アダプタの混在 { #mixing-2d-and-3d-moe-lora-adapters }

2D 形式（`megatron` ベース）と 3D 形式（`peft` ベース）のアダプタを同じエンジンインスタンスからサービングするには、`--enable-mixed-moe-lora-format` を付けてサーバーを起動し、各アダプタのレイアウトを `is_3d_lora_weight` フィールドで明示的に宣言します。

サーバー起動時（静的なモジュール指定）:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
    --enable-lora \
    --enable-mixed-moe-lora-format \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --lora-modules \
        '{"name": "lora-2d", "path": "jeeejeee/qwen36-35ba3b-2d-weights-poken-lora", "is_3d_lora_weight": false}' \
        '{"name": "lora-3d", "path": "jeeejeee/qwen36-35ba3b-moe-all-linear-poken-lora", "is_3d_lora_weight": true}'
```

`/v1/load_lora_adapter` による動的な読み込み:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
-H "Content-Type: application/json" \
-d '{
    "lora_name": "lora-3d",
    "lora_path": "/path/to/3d-format-lora",
    "is_3d_lora_weight": true
}'
```

!!! warning "アダプタのレイアウトを把握しておく必要があります"
    `--enable-mixed-moe-lora-format` を指定した場合、vLLM は呼び出し側が宣言した
    `is_3d_lora_weight` をそのまま信頼します。チェックポイントを検査して検証することは
    **ありません**。宣言が誤っていると、重みが誤ったスタックバッファへ読み込まれ、
    読み込み時にエラーも出ないまま、静かに無意味な出力を生成します。サービング前に
    レイアウトを確認してください。

    - **2D（エキスパートごと、megatron 形式）** → `is_3d_lora_weight: false` を設定します。
      アダプタのキーは `...experts.{idx}.gate_proj.lora_A.weight`、
      `...experts.{idx}.up_proj.lora_A.weight`、
      `...experts.{idx}.down_proj.lora_A.weight` のようになり、エキスパートごとに 1 組あります。
    - **3D（融合済み、peft 形式）** → `is_3d_lora_weight: true` を設定します。
      アダプタのキーは `...experts.gate_up_proj.lora_A.weight`、
      `...experts.down_proj.lora_A.weight` のようになり、全エキスパートを先頭次元に
      積み上げた単一のテンソルになります。

`--enable-mixed-moe-lora-format` を指定**しない**場合、`is_3d_lora_weight` は無視されます。
vLLM はベースモデルの `is_3d_moe_weight` からラッパーを選択し、アダプタ側がそれに一致している
必要があります。このフィールドは MoE でないモデルでも無視されます。

## モデルカードにおける LoRA モデルの系譜 { #lora-model-lineage-in-model-card }

`--lora-modules` の新しい形式は、主にモデルカードで親モデルの情報を表示できるようにするためのものです。現在のレスポンスがこれをどうサポートするかを説明します。

- LoRA モデル `sql-lora` の `parent` フィールドが、ベースモデル `meta-llama/Llama-3.2-3B-Instruct` を指すようになりました。これはベースモデルと LoRA アダプタの階層関係を正しく反映しています。
- `root` フィールドは、LoRA アダプタの成果物の場所を指します。

??? console "コマンドの出力"

    ```bash
    $ curl http://localhost:8000/v1/models

    {
        "object": "list",
        "data": [
            {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "meta-llama/Llama-3.2-3B-Instruct",
            "parent": null,
            "permission": [
                {
                .....
                }
            ]
            },
            {
            "id": "sql-lora",
            "object": "model",
            "created": 1715644056,
            "owned_by": "vllm",
            "root": "jeeejeee/llama32-3b-text2sql-spider",
            "parent": "meta-llama/Llama-3.2-3B-Instruct",
            "permission": [
                {
                ....
                }
            ]
            }
        ]
    }
    ```

## マルチモーダルモデルの Tower / Connector に対する LoRA サポート { #lora-support-for-tower-and-connector-of-multi-modal-model }

現在、vLLM はマルチモーダルモデルの Tower および Connector コンポーネントに対する LoRA を実験的にサポートしています。この機能を有効にするには、tower と connector に対応する token helper 関数を実装する必要があります。このアプローチの背景については [PR 26674](https://github.com/vllm-project/vllm/pull/26674) を参照してください。他のモデルの tower / connector への LoRA サポート拡張に向けたコントリビューションを歓迎します。現在のモデルのサポート状況は [Issue 31479](https://github.com/vllm-project/vllm/issues/31479) を参照してください。

## マルチモーダルモデル向けの既定 LoRA モデル { #default-lora-models-for-multimodal-models }

[Granite Speech](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) や [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) のような一部のモデルは、特定のモダリティが含まれるときに常に適用されることを前提とした LoRA アダプタを持ちます。これを上記の方法で管理するのはやや煩雑です。ユーザーが（オフラインでは）`LoRARequest` を送るか、（サーバーでは）リクエストのマルチモーダルデータの内容に応じてベースモデルと LoRA モデルへのリクエストを振り分ける必要があるためです。

そこで、既定のマルチモーダル LoRA を登録して自動的に処理できるようにしています。各モダリティを LoRA アダプタに対応づけておくと、対応する入力が含まれるときに自動的に適用されます。なお現時点では、1 プロンプトにつき LoRA は 1 つだけです。複数のモダリティが与えられ、そのそれぞれが何らかのモダリティに登録されている場合、いずれも適用されません。

??? code "オフライン推論での使用例"

    ```python
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.assets.audio import AudioAsset

    model_id = "ibm-granite/granite-speech-3.3-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_prompt(question: str, has_audio: bool):
        """Build the input prompt to send to vLLM."""
        if has_audio:
            question = f"<|audio|>{question}"
        chat = [
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False)


    llm = LLM(
        model=model_id,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=2048,
        limit_mm_per_prompt={"audio": 1},
        # Will always pass a `LoRARequest` with the `model_id`
        # whenever audio is contained in the request data.
        default_mm_loras = {"audio": model_id},
        enforce_eager=True,
    )

    question = "can you transcribe the speech into a written format?"
    prompt_with_audio = get_prompt(
        question=question,
        has_audio=True,
    )
    audio = AudioAsset("mary_had_lamb").audio_and_sample_rate

    inputs = {
        "prompt": prompt_with_audio,
        "multi_modal_data": {
            "audio": audio,
        }
    }


    outputs = llm.generate(
        inputs,
        sampling_params=SamplingParams(
            temperature=0.2,
            max_tokens=64,
        ),
    )
    ```

モダリティと LoRA モデル ID の対応を表す JSON 辞書を `--default-mm-loras` に渡すこともできます。たとえばサーバー起動時は次のようにします。

```bash
vllm serve ibm-granite/granite-speech-3.3-2b \
    --max-model-len 2048 \
    --enable-lora \
    --default-mm-loras '{"audio":"ibm-granite/granite-speech-3.3-2b"}' \
    --max-lora-rank 64
```

注: 既定のマルチモーダル LoRA は、現時点では `.generate` と chat completions でのみ利用できます。

## 利用上のヒント { #using-tips }

### `max_lora_rank` の設定 { #configuring-max_lora_rank }

`--max-lora-rank` パラメータは、LoRA アダプタに許容される最大ランクを制御します。この設定はメモリ確保と性能に影響します。

- 使用予定のすべての LoRA アダプタのうち、**最大のランクに合わせて設定する**
- **大きくしすぎない** — 必要よりずっと大きな値を使うとメモリを浪費し、性能上の問題を招くことがあります

たとえば LoRA アダプタのランクが [16, 32, 64] であれば、256 ではなく `--max-lora-rank 64` を使います。

```bash
# Good: matches actual maximum rank
vllm serve model --enable-lora --max-lora-rank 64

# Bad: unnecessarily high, wastes memory
vllm serve model --enable-lora --max-lora-rank 256
```

### LoRA を特定のモジュールに限定する { #restricting-lora-to-specific-modules }

`--lora-target-modules` パラメータを使うと、デプロイ時に LoRA を適用するモデルのモジュールを限定できます。特定の層にだけ LoRA が必要な場合の性能チューニングに役立ちます。

```bash
# Apply LoRA only to output projection layers
vllm serve model --enable-lora --lora-target-modules o_proj

# Apply LoRA to multiple specific modules
vllm serve model --enable-lora --lora-target-modules o_proj qkv_proj down_proj
```

`--lora-target-modules` を指定しない場合、LoRA はモデル内のサポートされるすべてのモジュールに適用されます。このパラメータは、`o_proj`、`qkv_proj`、`gate_proj` のようなモジュール名のサフィックス（モジュール名の最後の要素）を受け取ります。
