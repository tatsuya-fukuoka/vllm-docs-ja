# セキュリティ { #security }

## ノード間通信 { #inter-node-communication }

複数ノードで vLLM をデプロイした場合、ノード間のすべての通信は**既定では保護されていません**。ノードを隔離されたネットワークに配置して保護する必要があります。対象は次のとおりです。

1. PyTorch Distributed の通信
2. KV キャッシュ転送の通信
3. テンソル並列・パイプライン並列・データ並列の通信

### ノード間通信の設定オプション { #configuration-options-for-inter-node-communications }

vLLM のノード間通信は次のオプションで制御します。

#### 1. **環境変数** { #1-environment-variables }

- `VLLM_HOST_IP`: vLLM のプロセスが通信に使う IP アドレスを設定します

#### 2. **KV キャッシュ転送の設定** { #2-kv-cache-transfer-configuration }

- `--kv-ip`: KV キャッシュ転送の通信に使う IP アドレス（既定: 127.0.0.1）
- `--kv-port`: KV キャッシュ転送の通信に使うポート（既定: 14579）

#### 3. **データ並列の設定** { #3-data-parallel-configuration }

- `data_parallel_master_ip`: データ並列のマスターの IP（既定: 127.0.0.1）
- `data_parallel_master_port`: データ並列のマスターのポート（既定: 29500）

### PyTorch Distributed についての注意 { #notes-on-pytorch-distributed }

vLLM は一部のノード間通信に PyTorch の分散機能を使っています。PyTorch Distributed の
セキュリティ上の考慮事項については [PyTorch のセキュリティガイド](https://github.com/pytorch/pytorch/security/policy#using-distributed-features)（英語）を参照してください。

PyTorch のセキュリティガイドの要点:

- PyTorch Distributed の機能は内部通信のみを想定している
- 信頼できない環境やネットワークでの利用は想定されていない
- 性能上の理由から認可のプロトコルは含まれていない
- メッセージは暗号化されずに送信される
- どこからの接続でも検証なしに受け入れる

## セキュリティ上の推奨事項 { #security-recommendations }

### 1. **ネットワークの分離** { #1-network-isolation }

- vLLM のノードは専用の隔離されたネットワークにデプロイする
- ネットワークセグメンテーションで不正アクセスを防ぐ
- 適切なファイアウォールのルールを設定する

### 2. **設定のベストプラクティス** { #2-configuration-best-practices }

- `VLLM_HOST_IP` は既定値のままにせず、必ず具体的な IP アドレスを設定する
- ノード間で必要なポートだけを許可するようファイアウォールを設定する

### 3. **アクセス制御** { #3-access-control }

- デプロイ環境への物理的・ネットワーク的なアクセスを制限する
- 管理用インターフェイスには適切な認証・認可を実装する
- すべてのシステムコンポーネントで最小権限の原則に従う

### 4. **メディア URL のドメインを制限する** { #4-restrict-domains-access-for-media-urls }

`--allowed-media-domains` を設定して、vLLM がメディア URL としてアクセスできるドメインを制限し、
サーバーサイドリクエストフォージェリ (SSRF) 攻撃を防いでください。
（例: `--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`）

この保護は、オンラインサービングの API（マルチモーダル入力）と**バッチランナー**（`vllm run-batch`）の
両方に適用されます。バッチの文字起こし・翻訳リクエストに含まれる `file_url` も同じ許可リストで検証されます。

ドメインを制限しない場合、悪意あるユーザーが次のような URL を指定できてしまいます。

- **内部サービスを狙う**: 内部ネットワークのエンドポイント、クラウドのメタデータサービス
  （`169.254.169.254` など）、公開を意図していないサービスへのアクセス（SSRF）。
- **過剰にリソースを消費させる**: 極端に大きなファイルや応答の遅いエンドポイントを指定し、
  サーバーに際限なくデータをダウンロードさせて、メモリ・ディスク・ネットワーク帯域を枯渇させる。

メディアの取得元として想定するドメインだけを明示的に許可することで、こうした悪用の
攻撃面を大きく減らせます。

また、ドメイン制限を回避するための HTTP リダイレクトを追わないよう、
`VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` の設定も検討してください。

### 5. **メディアのデコードサイズを制限する** { #5-restrict-media-decode-sizes }

圧縮されたメディアファイルは、デコード時に数ギガバイトのメモリに展開されることがあります。
vLLM はメモリ不足によるサービス妨害を防ぐため、デコードサイズの上限を設けています。

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `VLLM_MAX_IMAGE_PIXELS` | `178956970`（約 1 億 7900 万ピクセル） | デコード後の画像サイズの上限（ピクセル数）。これを超える画像は、ラスタ用のメモリを確保する前に拒否されます。既定値は PIL 組み込みの解凍爆弾しきい値の 2 倍（RGB で約 680 MB）に合わせています。 |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `25` | 音声ファイル 1 つあたりの最大サイズ（MB）。 |
| `VLLM_MAX_AUDIO_DECODE_DURATION_S` | `600` | デコード後の音声の最大長（秒）。圧縮音声が数ギガバイトの float32 PCM に展開されるのを防ぎます。 |

いずれも `0` にすると対応する制限が無効になります。信頼できないユーザーに公開する環境では、
リソース枯渇攻撃への保護がなくなるため**推奨しません**。

## セキュリティとファイアウォール: 公開された vLLM の保護 { #security-and-firewalls-protecting-exposed-vllm-systems }

vLLM は、安全でないネットワークサービスをプライベートネットワークに隔離できるよう設計されていますが、
依存パッケージや基盤フレームワークなど、vLLM が直接制御できないところで、すべてのネットワーク
インターフェイスで待ち受ける安全でないサービスが開かれることがあります。

特に大きな懸念は `torch.distributed` の利用です。vLLM は単一ホストでの実行時も含め、分散通信に
これを利用しています。vLLM が TCP による初期化を行うと（[PyTorch の TCP 初期化のドキュメント](https://docs.pytorch.org/docs/stable/distributed.html#tcp-initialization)を参照）、
PyTorch は既定ですべてのネットワークインターフェイスで待ち受ける `TCPStore` を作成します。
つまり、追加の保護がなければ、いずれかのネットワークインターフェイス経由でマシンに到達できる
ホストからこれらのサービスにアクセスできてしまいます。

**PyTorch の観点では、`torch.distributed` の利用はすべて既定で安全でないと考えるべきです。**
これは PyTorch チームが認識したうえで意図している挙動です。

### ファイアウォール設定の指針 { #firewall-configuration-guidance }

vLLM を保護する最善の方法は、ファイアウォールを丁寧に設定し、必要最小限のネットワーク面だけを
公開することです。多くの場合、次のようになります。

- **API サーバーが待ち受ける TCP ポート以外への着信接続をすべてブロックする。**

- 内部通信に使うポート（`torch.distributed` や KV キャッシュ転送など）には、信頼できるホストや
ネットワークからのみアクセスできるようにする。

- これらの内部ポートを、インターネットや信頼できないネットワークに絶対に公開しない。

具体的なファイアウォールの設定方法は、OS やアプリケーションプラットフォームのドキュメントを参照してください。

## API キー認証の限界 { #api-key-authentication-limitations }

### 概要 { #overview }

`--api-key` フラグ（または環境変数 `VLLM_API_KEY`）は vLLM の HTTP サーバーに認証を提供しますが、
**対象は `/v1` パス配下の OpenAI 互換 API エンドポイントと、同様に `/v2`・`/inference` のパス接頭辞に限られます**。
機微な操作を伴う他の多くのエンドポイントは、同じ HTTP サーバー上で認証なしに公開されています。

**重要:** vLLM へのアクセス保護を `--api-key` だけに頼らないでください。本番環境では追加のセキュリティ対策が必要です。

### 保護されるエンドポイント（API キーが必要） { #protected-endpoints-require-api-key }

`--api-key` を設定すると、次の `/v1` エンドポイントは Bearer トークンによる認証が必要になります。

- `/v1/models` - 利用可能なモデルの一覧
- `/v1/chat/completions` - チャットの補完
- `/v1/chat/completions/batch` - チャット補完のバッチ実行
- `/v1/chat/completions/render` - チャット補完リクエストのレンダリング
- `/v1/chat/completions/derender` - チャット補完リクエストのデレンダリング
- `/v1/completions` - テキストの補完
- `/v1/completions/render` - 補完リクエストのレンダリング
- `/v1/completions/derender` - 補完リクエストのデレンダリング
- `/v1/embeddings` - 埋め込みの生成
- `/v1/audio/transcriptions` - 音声の文字起こし
- `/v1/audio/translations` - 音声の翻訳
- `/v1/messages` - Anthropic 互換の messages API
- `/v1/messages/count_tokens` - Anthropic messages のトークン数カウント
- `/v1/responses` - レスポンスの作成
- `/v1/responses/{response_id}` - レスポンスの取得
- `/v1/responses/{response_id}/cancel` - レスポンスのキャンセル
- `/v1/score` - スコアリング API
- `/v1/rerank` - リランキング API
- `/v1/load_lora_adapter` - LoRA アダプタの読み込み（モデルの挙動を変えうる。`--enable-lora` と `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` が設定されている場合のみ利用可能）
- `/v1/unload_lora_adapter` - LoRA アダプタの解放（モデルの挙動を変えうる。`--enable-lora` と `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` が設定されている場合のみ利用可能）
- `/inference/v1/generate` - 補完の生成
- `/v2/embed` - Cohere Embed API
- `/v2/rerank` - Cohere Rerank API

### 保護されないエンドポイント（API キー不要） { #unprotected-endpoints-no-api-key-required }

次のエンドポイントは、`--api-key` を設定していても**認証を必要としません**。

**推論のエンドポイント:**

- `/invocations` - SageMaker 互換のエンドポイント（`/v1` エンドポイントと同じ推論処理へ振り分けられます）
- `/generative_scoring` - Generative scoring API
- `/pooling` - プーリング API
- `/classify` - 分類 API
- `/score` - スコアリング API（`/v1` 以外の系統）
- `/rerank` - リランキング API（`/v1` 以外の系統）

**運用制御のエンドポイント（`"generate"` タスクをサポートする場合のみ）:**

- `/pause` - 生成の一時停止（サービス停止状態になります）
- `/resume` - 生成の再開
- `/is_paused` - 生成が一時停止中かどうかの確認
- `/abort_requests` - 実行中リクエストの中断（実行中の処理が失われます）
- `/scale_elastic_ep` - スケーリング操作の実行
- `/is_scaling_elastic_ep` - スケーリング中かどうかの確認
- `/init_weight_transfer_engine` - RLHF 用の重み転送エンジンの初期化
- `/update_weights` - モデルの重みの更新（モデルの挙動を変えうる）
- `/get_world_size` - 分散実行の world size の取得
- `/abort_requests` - 実行中リクエストの中断（`--tokens-only` も指定した場合のみ）

**ユーティリティのエンドポイント:**

- `/tokenize` - テキストのトークナイズ
- `/detokenize` - トークンのデトークナイズ
- `/health` - ヘルスチェック
- `/ping` - SageMaker のヘルスチェック
- `/version` - バージョン情報
- `/load` - サーバー負荷のメトリクス

**トークナイザー情報のエンドポイント（`--enable-tokenizer-info-endpoint` を指定した場合のみ）:**

このエンドポイントは **`--enable-tokenizer-info-endpoint` フラグを指定した場合にのみ利用できます**。
チャットテンプレートやトークナイザーの設定など、機微な情報が露出する可能性があります。

- `/tokenizer_info` - チャットテンプレートや設定を含む、トークナイザーの詳細情報の取得

**開発用のエンドポイント（`VLLM_SERVER_DEV_MODE=1` の場合のみ）:**

これらのエンドポイントは**環境変数 `VLLM_SERVER_DEV_MODE` が `1` の場合にのみ利用できます**。
開発・デバッグ目的のものであり、本番環境では絶対に有効にしないでください。

- `/server_info` - サーバーの詳細な設定の取得
- `/reset_prefix_cache` - プレフィックスキャッシュのリセット（サービスに影響する可能性があります）
- `/reset_mm_cache` - マルチモーダルキャッシュのリセット（サービスに影響する可能性があります）
- `/reset_encoder_cache` - エンコーダーキャッシュのリセット（サービスに影響する可能性があります）
- `/sleep` - エンジンをスリープ状態にする（サービス停止状態になります）
- `/wake_up` - スリープ状態からの復帰
- `/is_sleeping` - エンジンがスリープ中かどうかの確認
- `/collective_rpc` - エンジン上での任意の RPC メソッドの実行（極めて危険）

**プロファイラのエンドポイント（`--profiler-config` でプロファイリングを有効にした場合のみ）:**

これらのエンドポイントはプロファイリングを有効にした場合にのみ利用でき、ローカル開発でのみ使うべきです。

- `/start_profile` - PyTorch プロファイラの開始
- `/stop_profile` - PyTorch プロファイラの停止

**注意:** 特に `/invocations` は、保護された `/v1` エンドポイントと同じ推論機能へ認証なしで
アクセスできてしまうため、注意が必要です。

### セキュリティ上の影響 { #security-implications }

vLLM の HTTP サーバーに到達できる攻撃者は、次のことが可能です。

1. `/invocations`、`/inference/v1/generate`、`/generative_scoring`、`/pooling`、`/classify`、`/score`、`/rerank` など `/v1` 以外のエンドポイントを使って**認証を回避**し、資格情報なしで任意の推論を実行する
2. トークンなしで `/pause`、`/scale_elastic_ep`、`/abort_requests` を呼び出して**サービス妨害を引き起こす**
3. **運用制御にアクセス**してサーバーの状態を操作する（生成の一時停止、`/update_weights` によるモデル重みの更新など）
4. **`--enable-tokenizer-info-endpoint` が設定されている場合:** チャットテンプレートを含むトークナイザー設定にアクセスし、プロンプト設計の戦略や実装の詳細を知る
5. **`VLLM_SERVER_DEV_MODE=1` が設定されている場合:** `/collective_rpc` で任意の RPC コマンドを実行し、キャッシュをリセットし、エンジンをスリープさせ、サーバーの詳細な設定にアクセスする

### 推奨されるセキュリティ対策 { #recommended-security-practices }

#### 1. 公開するエンドポイントを最小化する { #1-minimize-exposed-endpoints }

**重要:** 本番環境では `VLLM_SERVER_DEV_MODE=1` を絶対に設定しないでください。開発用エンドポイントは
次のような極めて危険な機能を露出します。

- `/collective_rpc` による任意の RPC 実行
- サービスに影響しうるキャッシュ操作
- サーバー設定の詳細な開示

同様に、本番環境ではプロファイラのエンドポイントも有効にしないでください。

**`--enable-tokenizer-info-endpoint` の扱いには注意してください:** トークナイザーの設定情報を公開する
必要がある場合にのみ `/tokenizer_info` を有効にしてください。このエンドポイントは、実装の詳細や
プロンプト設計の戦略を含みうるチャットテンプレートとトークナイザー設定を露出します。

#### 2. リバースプロキシの背後にデプロイする { #2-deploy-behind-a-reverse-proxy }

もっとも効果的なのは、次のようなリバースプロキシ（nginx、Envoy、Kubernetes Gateway など）の
背後に vLLM をデプロイすることです。

- エンドユーザーに公開したいエンドポイントだけを明示的に許可する
- 認証のない推論エンドポイントや運用制御のエンドポイントを含め、その他をすべてブロックする
- プロキシ層で追加の認証・レート制限・ログ記録を実装する

## リクエストパラメータによるリソース制限 { #request-parameter-resource-limits }

一部の API リクエストパラメータはリソース消費に大きく影響し、サーバーのリソースを枯渇させる目的で
悪用されるおそれがあります。`/v1/completions` と `/v1/chat/completions` の `n` パラメータは、
1 リクエストで生成する独立した出力系列の数を制御します。極端に大きな値を指定すると、エンジンは
`n` に比例したメモリ・CPU・GPU 時間を確保するため、ホストのメモリ不足を招き、他のリクエストの
処理を妨げる可能性があります。

これを緩和するため、vLLM は環境変数 `VLLM_MAX_N_SEQUENCES`（既定: **16384**）で `n` の上限を
設定できるようにしています。この上限を超えるリクエストは、エンジンに届く前に拒否されます。

### 推奨事項 { #recommendations }

- **一般公開する環境:** 1 リクエストの影響範囲を抑えるため、ワークロードに見合った値
  （`64` や `128` など）を `VLLM_MAX_N_SEQUENCES` に設定することを検討してください。
- **リバースプロキシ層:** vLLM 側の上限に加えて、リバースプロキシでリクエストボディの検証や
  レート制限を行い、悪意あるペイロードをさらに制限することを検討してください。
- **監視:** リクエストごとのリソース消費を監視し、悪用の兆候となる異常なパターンを検知してください。

## ツールサーバーと MCP のセキュリティ { #tool-server-and-mcp-security }

vLLM は `--tool-server` 引数で外部のツールサーバーに接続できます。これにより、モデルは
Responses API (`/v1/responses`) を通じてツールを呼び出せます。ツールサーバーのサポートは
すべてのモデルで機能し、特定のモデルアーキテクチャに限定されません。

**重要:** ツールサーバーは既定では 1 つも有効になっていません。設定で明示的に有効にする必要があります。

### 組み込みのデモ用ツール (GPT-OSS) { #built-in-demo-tools-gpt-oss }

`--tool-server demo` を指定すると、ツール呼び出しに対応した任意のモデルで使える組み込みのデモ用
ツールが有効になります。ツールの実装は vLLM の一部ではなく、別途インストールする
[`gpt-oss`](https://github.com/openai/gpt-oss) パッケージが提供します。vLLM は `gpt-oss` に処理を
委譲する薄いラッパーを提供しているだけです。

- **コードインタプリタ** (`python`): Docker 経由の Python 実行（`gpt_oss.tools.python_docker`）
- **Web ブラウザ** (`browser`): Exa API による検索。`EXA_API_KEY` が必要（`gpt_oss.tools.simple_browser`）

#### コードインタプリタ（Python ツール）のセキュリティリスク { #code-interpreter-python-tool-security-risks }

コードインタプリタは、モデルが生成したコードを Docker コンテナ内で実行します。ただし、この
コンテナは**既定ではネットワークが分離されていません**。ホストの Docker のネットワーク設定
（既定のブリッジネットワークや `--network=host` など）を引き継ぐため、次のような状況が起こりえます。

- コンテナからホストのネットワークや LAN にアクセスできる可能性がある。
- コンテナから到達できる内部サービスが SSRF（サーバーサイドリクエストフォージェリ）で悪用される可能性がある。
- クラウドのメタデータサービス（`169.254.169.254` など）にアクセスできる可能性がある。
- `torch.distributed` のエンドポイントなど、脆弱な内部サービスにコンテナから到達できる場合、それらへの攻撃に使われる可能性がある。

実行されるコードがモデルによって生成される、つまり敵対的な入力（プロンプトインジェクション）の
影響を受けうるという点で、特に注意が必要です。

#### 組み込みツールの有効・無効の制御 { #controlling-built-in-tool-availability }

組み込みのデモ用ツールは 2 つの設定で制御します。

1. **`--tool-server demo`**: 組み込みのデモ用ツール（ブラウザと Python コードインタプリタ）を有効にします。

2. **`VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`**: Responses API の `mcp` ツールタイプ経由で組み込みツールが
   要求されたとき、どのツールラベルを許可するかをカンマ区切りの許可リストで指定します。有効な値:
   - `container` - コンテナツール
   - `code_interpreter` - Python コード実行ツール
   - `web_search_preview` - Web 検索・ブラウザツール

   この変数が未設定または空の場合、MCP ツールタイプ経由で要求された組み込みツールは有効になりません。

Python のコードインタプリタだけを無効にしたい場合は、`VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS` から
`code_interpreter` を外してください。

**独自実装の検討**: GPT-OSS の Python ツールは参照実装です。本番環境では、より厳格な分離を保証する
独自のコード実行サンドボックスの実装を検討してください。指針は [GPT-OSS のドキュメント](https://github.com/openai/gpt-oss?tab=readme-ov-file#python)（英語）を参照してください。

## LoRA の動的ロード { #dynamic-lora-loading }

vLLM は `/v1/load_lora_adapter` と `/v1/unload_lora_adapter` の API エンドポイントを通じて、実行時に
LoRA アダプタを動的にロード・アンロードできます。この機能は**既定では無効**で、`--enable-lora` と
環境変数 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` の両方を設定する必要があります。

**警告:** LoRA の動的ロードは安全な操作ではなく、信頼できないクライアントに公開する環境では
有効にすべきではありません。どうしても有効にする必要がある場合は、リバースプロキシや
ネットワークレベルのアクセス制御を用いて、`/v1/load_lora_adapter` と `/v1/unload_lora_adapter` への
アクセスを信頼できる管理者のみに制限してください。エンドユーザーには公開しないでください。
LoRA アダプタの設定方法は [LoRA アダプタのドキュメント](../features/lora.md)を参照してください。

## エンドポイントプラグイン { #endpoint-plugins }

vLLM は `vllm.endpoint_plugins` のエントリポイントグループを通じて、ツリー外の HTTP ルートを
読み込めます（書き方は[エンドポイントプラグイン](../design/endpoint_plugins.md)を参照）。
エンドポイントプラグインは任意の FastAPI ルートを登録でき、`EngineClient.collective_rpc` 経由で
エンジンに到達するルートも登録できるため、サンドボックス化された入力ではなく、サーバーの
信頼されたコードベースの一部として扱う必要があります。

**エンドポイントプラグインは既定では読み込まれません。** 他の vLLM プラグイングループ
（`vllm.general_plugins`、`vllm.platform_plugins` など）は `VLLM_PLUGINS` で絞り込まない限り
発見したプラグインをすべて読み込みますが、エンドポイントプラグインは `VLLM_PLUGINS` で明示的に
名前を指定しない限り**まったく読み込まれません**。これは `VLLM_SERVER_DEV_MODE` で保護された
開発用エンドポイントと同じ「本番では既定で無効」という方針に沿ったものです。どちらの機能も、
運用者が明示的に選択した場合にのみ現れます。

### 推奨されるセキュリティ対策 { #recommended-security-practices_1 }

1. **信頼するプラグインだけを許可リストに入れる。** `VLLM_PLUGINS` には実行する意図のあるプラグイン名を正確に設定し、ワイルドカードを使ったり、各プラグインの内容を確認せずに許可リストを別環境からコピーしたりしないでください。
2. **デプロイ前にルートを監査する。** プラグインの `attach_router` は、既存の `/v1/*` と重複するものを含め、任意のパスにルートを追加できます。現時点でルート衝突の検出は行われていないため（RFC [#46565](https://github.com/vllm-project/vllm/issues/46565) のフォローアップとして追跡中）、悪意ある、あるいはバグのあるプラグインが**コアのルートを覆い隠し**、その挙動を黙って置き換えてしまう可能性があります。`/v1/...` を再利用せず、独自の接頭辞（`/plugins/<plugin-name>/...` など）でルートを名前空間化するプラグインを選び、実際に何が提供されているか確実に把握したい場合は起動後に `app.routes` を確認してください。
3. **プラグインのルートも「既定で認証なし」の面として扱う。** `--api-key` が保護するのは `/v1`、`/v2`、`/inference` のパス接頭辞だけです（[API キー認証の限界](#api-key-authentication-limitations)を参照）。これらの接頭辞の外にあるプラグインのルートは、プラグイン自身が認証を実装していない限り認証されません。外部に公開したいプラグインのルートだけを許可するリバースプロキシの背後にデプロイしてください。
4. **`vllm.general_plugins` との対応関係に注意する。** エンジン側の新しい挙動も必要とするプラグインは、その半分を `vllm.general_plugins` として別途提供します。こちらは既定の方針（制限しない限りすべて読み込む）に従い、すべてのワーカープロセスで読み込まれます。エンドポイントプラグインを許可リストで制限しても、対になるエンジン側プラグインは制限されません。両方を確認する必要があります。

## gRPC インターフェイス { #grpc-interface }

vLLM は `--grpc-port` フラグで有効にできる、別 TCP ポートのオプションの gRPC Generate サービスを
提供します。指定しない場合、gRPC サーバーは起動しません。gRPC のリスナーは HTTP サーバーと
同じホストアドレスにバインドされます。

**警告:** gRPC インターフェイスは**既定で保護されていません**。認証・認可・暗号化のいずれも実装
されていません。信頼されたネットワーク内で同居するサービス間でのみ使う、プライベートな内部
インターフェイスとみなしてください。gRPC ポートをインターネットや信頼できないクライアントに
公開しないでください。有効にする場合は、ファイアウォールのルール、ネットワークセグメンテーション、
隔離されたプライベートネットワークへの配置など、ネットワークレベルのアクセス制御で保護してください。

### セキュリティ上の影響 { #security-implications_1 }

gRPC ポートに到達できる攻撃者は、次のことが可能です。

1. `Generate` と `GenerateStream` の RPC を使い、資格情報なしで**任意の推論を実行する**
2. 際限のない生成リクエストを送りつけて **GPU と計算リソースを消費させる**
3. gRPC インターフェイスのバグを突いて vLLM をクラッシュさせ、**サービス妨害を引き起こす**

### 推奨事項 { #recommendations_1 }

- gRPC による推論が明確に必要な場合にのみ `--grpc-port` を有効にする
- gRPC ポートには信頼できるホストやサービスからのみアクセスできるようにする
- ファイアウォールのルールで gRPC ポートへの外部アクセスをブロックする
- gRPC インターフェイスを専用の内部ネットワークインターフェイスに配置することを検討する

## キャッシュディレクトリのセキュリティ { #cache-directory-security }

vLLM は、キャッシュディレクトリが**プライベートかつ信頼できる**ことを前提としています。キャッシュの
内容は暗号学的な完全性検証なしに読み込まれ、その中には任意コード実行が可能な形式も含まれます。
信頼できないユーザーやプロセスが vLLM のキャッシュディレクトリに書き込めると、vLLM を
クラッシュさせたり、任意のコードを実行させたりできる可能性があります。

**vLLM のキャッシュディレクトリを信頼できないユーザーと共有したり、信頼できないストレージから
マウントしたりしないでください。** キャッシュディレクトリは vLLM のインストール自体と同じ注意を
払って扱ってください。

### キャッシュディレクトリの設定 { #cache-directory-configuration }

ほとんどのキャッシュパスは単一のルート配下のサブディレクトリを既定値としています。
`VLLM_CACHE_ROOT` を変更すると、そこから派生するすべての機能の既定の場所が変わります。
`torch.compile` のキャッシュが有効な場合（既定）、vLLM は `TRITON_CACHE_DIR` もこのツリー内に
向けます。コンパイルキャッシュを無効にした場合、Triton は自身の既定の場所（`~/.triton/cache`）に
戻ります。

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | キャッシュの基準ディレクトリ。設定されていれば `XDG_CACHE_HOME` を尊重します。明示的に上書きしない限り、以下のパスはすべてここから派生します。 |
| *(torch.compile)* | `$VLLM_CACHE_ROOT/torch_compile_cache/` | AOT コンパイル済みモデル、Inductor のグラフ、Triton のカーネルのコンパイルキャッシュ。`VLLM_DISABLE_COMPILE_CACHE` で制御します（`1` で無効）。 |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `$VLLM_CACHE_ROOT/flashinfer_autotune_cache/<flashinfer-version>/<arch>/<cache-hash>/` | FlashInfer のオートチューニング設定のキャッシュ。 |
| `VLLM_ASSETS_CACHE` | `$VLLM_CACHE_ROOT/assets/` | ダウンロードしたアセット（トークナイザーのファイルなど）。 |
| `VLLM_XLA_CACHE_PATH` | `$VLLM_CACHE_ROOT/xla_cache/` | XLA / TPU のコンパイルキャッシュ。 |
| `VLLM_MEDIA_CACHE` | *(無効)* | ダウンロードしたメディア（画像・動画・音声）の任意のキャッシュ。明示的に設定しない限り有効になりません。 |

### 推奨事項 { #recommendations_2 }

- `VLLM_CACHE_ROOT`（およびコンパイルキャッシュを無効にしている場合の `~/.triton` など、依存パッケージが使う他のキャッシュディレクトリ）の**ファイルパーミッションを制限**し、vLLM プロセスの所有者だけが読み書きできるようにする。
- **信頼できない場所からキャッシュの内容をコピーしない。** 環境間でキャッシュの成果物を配布する場合は、信頼できるビルドパイプライン由来であることを確認してください。
- **コンテナでのデプロイ:** キャッシュディレクトリをコンテナにマウントする場合は、ボリュームの元が信頼できることを確認してください。

## FIPS 互換性 { #fips-compatibility }

FIPS 準拠は多くの要因に左右されるため、vLLM のデプロイが自動的に FIPS 準拠になることはありません。
近年の変更で、FIPS が有効なホストに対する vLLM の*耐性*は向上しました。つまり、承認されていない
アルゴリズムがブロックされてもクラッシュしなくなりました。ただし耐性と準拠は別物です。デプロイが
FIPS の要件を満たすかどうかは、ホスト OS、Python の `hashlib` と `ssl` を支える OpenSSL プロバイダ、
インストールされている任意依存パッケージに依存します。

### FIPS に関わる設定 { #fips-relevant-configuration }

FIPS が有効なホストで vLLM を運用する場合、次の設定で FIPS 承認済みのアルゴリズムを選択してください。

- **マルチモーダル入力のハッシュ** — `VLLM_MM_HASHER_ALGORITHM` の既定値は `blake3` で、FIPS 承認されていません。FIPS 環境では `sha256` または `sha512` を設定してください。
- **プレフィックスキャッシュのハッシュ** — `--prefix-caching-hash-algo`（設定項目 `prefix_caching_hash_algo`）に `sha256` または `sha256_cbor` を設定してください。`xxhash` と `xxhash_cbor` は FIPS 承認されていません。
- **TLS の暗号スイート** — `--ssl-ciphers` を使い、API サーバーの TLS ハンドシェイクを環境のポリシーに合った FIPS 承認済みの暗号スイートに制限してください。

### セキュリティ目的でない MD5 利用の自動フォールバック { #automatic-fallback-for-non-security-md5-use }

vLLM は、セキュリティ目的でないキャッシュキーの導出（設定のハッシュなど）に数箇所で MD5 を使って
います。これらの呼び出しは `usedforsecurity=False` を渡し、さらに基盤の OpenSSL プロバイダが MD5 を
完全に拒否する場合には SHA-256 にフォールバックします（`vllm/utils/hashing.py` の `safe_hash()` を参照）。
利用者側の対応は不要ですが、監査担当やセキュリティレビュー担当が MD5 の参照箇所を特定し、その目的を
理解できるようここに記載しています。

### FIPS 非承認のハッシュ実装を提供する依存パッケージ { #dependencies-that-provide-non-fips-hash-implementations }

一部の依存パッケージは FIPS 承認されていないハッシュ実装を提供します。vLLM は該当するアルゴリズムが
選択されたときにのみそれらを呼び出しますが、厳格な暗号統制を求められる運用者は、そのコードパスが
実行されないこと、さらにポリシーによってはパッケージ自体が存在しないことを確認したい場合があります。

- `blake3` — 現在 `requirements/common.txt` に含まれているため、通常のインストールで導入されます。遅延インポートされ、`VLLM_MM_HASHER_ALGORITHM=blake3`（既定）のときにのみ使われます。`VLLM_MM_HASHER_ALGORITHM` を `sha256` か `sha512` にすれば、FIPS 非承認のコードパスは実行されません。ポリシーがパッケージの存在自体を禁じている場合は、`pip install` 後に削除してください（`pip uninstall blake3`）。`VLLM_MM_HASHER_ALGORITHM` が blake3 以外であれば vLLM は問題なく動作します。
- `xxhash` — 完全に任意の依存パッケージです（`requirements/common.txt` には含まれません）。`xxhash` ベースのプレフィックスキャッシュのアルゴリズムを選択したときにのみインポートされます。インストールせず、`sha256` ベースのアルゴリズムを選択してください。

### ハッシュ以外の FIPS 上の考慮事項 { #beyond-hashing-other-fips-considerations }

vLLM が FIPS を意識したコードを持つのはハッシュの領域ですが、FIPS 準拠のデプロイは vLLM の外側にある
複数の要素に依存します。運用者はプラットフォームとセキュリティのチームとともに次を評価してください。

- **ホストの暗号プロバイダ。** Python の `hashlib` と `ssl` が FIPS を意識するのは、Python がホスト OS の提供する FIPS 検証済み OpenSSL（または同等品）にリンクされている場合だけです。vLLM はホストが設定したプロバイダをそのまま利用し、自前では同梱しません。
- **API サーバーの TLS。** OpenAI 互換 API サーバーの TLS 終端は、Python の `ssl` モジュール経由でホストの OpenSSL を使います。環境の FIPS ポリシーに合わせて `--ssl-ciphers` で暗号スイートを制限し、サーバー証明書が FIPS 承認済みのアルゴリズムと鍵長で発行されていることを確認してください。
- **外向きの HTTPS。** モデルやアセットのダウンロード（`huggingface_hub` 経由など）も同じホストの TLS スタックを使います。プロバイダと暗号スイートに関する考慮は同様です。
- **ノード間通信は既定で暗号化されません。** [ノード間通信](#inter-node-communication)で述べたとおり、PyTorch Distributed、KV キャッシュ転送、データ並列の通信は暗号化されません。転送中データに FIPS 承認済みの暗号を要求する環境では、mTLS のサイドカーや FIPS 検証済みモジュールで終端する IPsec など、外部で保護を提供する必要があります。vLLM の内部チャネル単体ではこの要件を満たせません。ネットワークの分離は暗号ではないため「転送中データの FIPS 承認済み暗号」という要件は満たしませんが、多層防御としては有用です。
- **独自の OpenSSL を同梱する依存パッケージ。** 一部の Python wheel は、FIPS が有効なホストでカーネルの FIPS セルフテストに失敗する OpenSSL を静的リンクしています（`FATAL FIPS SELFTEST FAILURE`）。`opencv-python-headless` は既知の例で、他の manylinux wheel も同様の挙動を示すことがあります。FIPS 環境での起動失敗を調査する際は、インストール済み wheel に暗号ライブラリが同梱されていないか確認してください。
- **アクセラレータと ML ライブラリ。** PyTorch、CUDA、cuDNN、NCCL などのコンポーネントは、vLLM とは独立した暗号と FIPS の状況を持ちます。NVIDIA は一部のライブラリについて FIPS 検証済みビルドを公開していますが、vLLM はそれらに固定していないため、選定と検証は運用者の責任です。
- **vLLM において FIPS の対象*外*であるもの。** トークンのサンプリングに使う乱数生成（Python / NumPy / PyTorch の RNG）は暗号用途ではないため、FIPS の対象外です。pickle 化されたキャッシュの成果物は別のセキュリティ上の論点であり、[キャッシュディレクトリのセキュリティ](#cache-directory-security)で扱っています。
