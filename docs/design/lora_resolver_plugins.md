# LoRA リゾルバプラグイン { #lora-resolver-plugins }

このディレクトリには、`LoRAResolver` フレームワークをもとに構築された vLLM の LoRA リゾルバプラグインが含まれています。これらのプラグインは、指定されたローカルストレージのパスから LoRA アダプターを自動的に検出して読み込むため、手動での設定やサーバーの再起動が不要になります。

## 概要 { #overview }

LoRA リゾルバプラグインは、実行時に LoRA アダプターを動的に読み込むための柔軟な手段を提供します。まだ読み込まれていない LoRA アダプターへのリクエストを vLLM が受け取ると、リゾルバプラグインが設定済みのストレージからそのアダプターを探し出して読み込もうとします。これにより次のことが可能になります。

- **LoRA の動的読み込み**: サーバーを再起動せずにオンデマンドでアダプターを読み込む
- **複数のストレージバックエンド**: ファイルシステム、S3、独自バックエンドに対応。組み込みの `lora_filesystem_resolver` はローカルストレージのパスを必要とし、組み込みの `hf_hub_resolver` は Huggingface Hub から LoRA アダプターを取得して同様に処理します。一般に、任意のソースから取得するカスタムリゾルバを実装できます。
- **自動検出**: 既存の LoRA ワークフローとのシームレスな統合
- **スケーラブルなデプロイ**: 複数の vLLM インスタンスにまたがるアダプターの集中管理

## 前提条件 { #prerequisites }

LoRA リゾルバプラグインを使う前に、次の環境変数が設定されていることを確認してください。

### 必須の環境変数 { #required-environment-variables }

1. **`VLLM_ALLOW_RUNTIME_LORA_UPDATING`**: LoRA の動的読み込みを有効にするため、`true` または `1` に設定する必要があります
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   ```

2. **`VLLM_PLUGINS`**: 使用したいリゾルバプラグインを含める必要があります（カンマ区切りのリスト）
   ```bash
   export VLLM_PLUGINS=lora_filesystem_resolver
   ```

3. **`VLLM_LORA_RESOLVER_CACHE_DIR`**: ファイルシステムリゾルバ用に、有効なディレクトリパスを設定する必要があります
   ```bash
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

### 任意の環境変数 { #optional-environment-variables }

- **`VLLM_PLUGINS`**: 設定されていない場合、利用可能なすべてのプラグインが読み込まれます。空文字列を設定すると、プラグインは一切読み込まれません。

## 利用可能なリゾルバ { #available-resolvers }

### lora_filesystem_resolver { #lora_filesystem_resolver }

ファイルシステムリゾルバは既定で vLLM と一緒にインストールされ、ローカルのディレクトリ構造から LoRA アダプターを読み込めるようにします。

#### セットアップ手順 { #setup-steps }

1. **LoRA アダプターの保存ディレクトリを作成する**:
   ```bash
   mkdir -p /path/to/lora/adapters
   ```

2. **環境変数を設定する**:
   ```bash
   export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
   export VLLM_PLUGINS=lora_filesystem_resolver
   export VLLM_LORA_RESOLVER_CACHE_DIR=/path/to/lora/adapters
   ```

3. **vLLM サーバーを起動する**:
   ベースモデルには `meta-llama/Llama-2-7b-hf` などを使えます。環境変数に Hugging Face のトークンを設定しておいてください（`export HF_TOKEN=xxx235`）。
   ```bash
   vllm serve your-base-model \
       --enable-lora
   ```

#### ディレクトリ構造の要件 { #directory-structure-requirements }

ファイルシステムリゾルバは、LoRA アダプターが次の構造で配置されていることを前提とします。

```text
/path/to/lora/adapters/
├── adapter1/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (if applicable)
├── adapter2/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files (if applicable)
└── ...
```

各アダプターのディレクトリには、次のものが必要です。

- **`adapter_config.json`**: 次の構造を持つ必須の設定ファイル:
  ```json
  {
    "peft_type": "LORA",
    "base_model_name_or_path": "your-base-model-name",
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "modules_to_save": null,
    "use_rslora": false,
    "use_dora": false
  }
  ```

- **`adapter_model.bin`**: LoRA アダプターの重みファイル

#### 使用例 { #usage-example }

1. **LoRA アダプターを用意する**:
   ```bash
   # Assuming you have a LoRA adapter in /tmp/my_lora_adapter
   cp -r /tmp/my_lora_adapter /path/to/lora/adapters/my_sql_adapter
   ```

2. **ディレクトリ構造を確認する**:
   ```bash
   ls -la /path/to/lora/adapters/my_sql_adapter/
   # Should show: adapter_config.json, adapter_model.bin, etc.
   ```

3. **アダプターを使ってリクエストを送る**:
   ```bash
   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "my_sql_adapter",
           "prompt": "Generate a SQL query for:",
           "max_tokens": 50,
           "temperature": 0.1
       }'
   ```

#### 動作の仕組み { #how-it-works }

1. vLLM が `my_sql_adapter` という名前の LoRA アダプターへのリクエストを受け取る
2. ファイルシステムリゾルバが `/path/to/lora/adapters/my_sql_adapter/` の存在を確認する
3. 見つかった場合、`adapter_config.json` を検証する
4. 設定がベースモデルと一致し、内容が妥当であれば、アダプターが読み込まれる
5. 新しく読み込まれたアダプターを使って、リクエストが通常どおり処理される
6. 以降のリクエストでも、そのアダプターは引き続き利用できる

## 高度な設定 { #advanced-configuration }

### 複数のリゾルバ { #multiple-resolvers }

異なるソースからアダプターを読み込むために、複数のリゾルバプラグインを設定できます。

`lora_s3_resolver` は、自分で実装する必要があるカスタムリゾルバの例です。

```bash
export VLLM_PLUGINS=lora_filesystem_resolver,lora_s3_resolver
```

列挙したリゾルバはすべて有効になります。リクエスト時、vLLM は成功するまで順番に試します。

### カスタムリゾルバの実装 { #custom-resolver-implementation }

独自のリゾルバプラグインを実装するには次のようにします。

1. **新しいリゾルバクラスを作成する**:
   ```python
   from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
   from vllm.lora.request import LoRARequest
   
   class CustomResolver(LoRAResolver):
       async def resolve_lora(self, base_model_name: str, lora_name: str) -> Optional[LoRARequest]:
           # Your custom resolution logic here
           pass
   ```

2. **リゾルバを登録する**:
   ```python
   def register_custom_resolver():
       resolver = CustomResolver()
       LoRAResolverRegistry.register_resolver("Custom Resolver", resolver)
   ```

## トラブルシューティング { #troubleshooting }

### よくある問題 { #common-issues }

1. **"VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory"**
   - ディレクトリが存在し、アクセス可能であることを確認してください
   - ディレクトリのファイル権限を確認してください

2. **"LoRA adapter not found"**
   - アダプターのディレクトリ名が、リクエストしたモデル名と一致しているか確認してください
   - `adapter_config.json` が存在し、妥当な JSON であることを確認してください
   - ディレクトリに `adapter_model.bin` が存在することを確認してください

3. **"Invalid adapter configuration"**
   - `peft_type` が "LORA" になっているか確認してください
   - `base_model_name_or_path` がベースモデルと一致しているか確認してください
   - `target_modules` が正しく設定されているか確認してください

4. **"LoRA rank exceeds maximum"**
   - `adapter_config.json` の `r` の値が `max_lora_rank` の設定を超えていないか確認してください

### デバッグのヒント { #debugging-tips }

1. **デバッグログを有効にする**:
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   ```

2. **環境変数を確認する**:
   ```bash
   echo $VLLM_ALLOW_RUNTIME_LORA_UPDATING
   echo $VLLM_PLUGINS
   echo $VLLM_LORA_RESOLVER_CACHE_DIR
   ```

3. **アダプターの設定をテストする**:
   ```bash
   python -c "
   import json
   with open('/path/to/lora/adapters/my_adapter/adapter_config.json') as f:
       config = json.load(f)
   print('Config valid:', config)
   "
   ```
