# スリープモード { #sleep-mode }

vLLM のスリープモードを使うと、サーバーを停止したり Docker コンテナを終了したりすることなく、モデルの重みや KV キャッシュを含む GPU メモリの大部分を一時的に解放できます。RLHF や学習、あるいは推論ワークロードの合間に GPU リソースを解放してコストを抑えたい場面で特に役立ちます。

主な利点:

- **GPU メモリを解放**: モデルの重みを CPU RAM に退避し、KV キャッシュを破棄することで、GPU メモリの 90% 以上を他のタスク向けに解放できます。
- **高速な再開**: モデルを完全に読み込み直すことなく、エンジンをすばやく起こして推論を再開できます。
- **API エンドポイント**: sleep / wake_up の状態を HTTP エンドポイントまたは Python API から制御できます。
- **分散ワークロードに対応**: テンソル並列やパイプライン並列などと併用できます。
- **細かい制御**: モデルの重みだけ、または KV キャッシュだけを起こすこともでき、重み更新中の OOM を避けられます。

!!! note
    この機能は現在、CUDA と ROCm のプラットフォームでサポートされています。

!!! note
    詳細はこちらの[ブログ記事](https://blog.vllm.ai/2025/10/26/sleep-mode.html)を参照してください。

## スリープのレベル { #sleep-levels }

レベル 1 のスリープでは、モデルの重みを退避し、KV キャッシュを破棄します。KV キャッシュの内容は失われます。レベル 1 のスリープは、同じモデルを再び動かすためにエンジンをスリープさせて起こす用途に適しています。モデルの重みは CPU メモリにバックアップされるため、重みを保持できるだけの CPU メモリがあることを確認してください。レベル 2 のスリープでは、モデルの重みと KV キャッシュの両方を破棄します（rope スケーリングのテンソルなど、モデルのバッファは CPU 上に保持されます）。モデルの重みと KV キャッシュの内容はどちらも失われます。レベル 2 のスリープは、以前のモデルの重みが不要な場合、たとえば RLHF の重み更新のように、別のモデルを動かしたりモデルを更新したりするためにエンジンをスリープさせて起こす用途に適しています。

## 使い方 { #usage }

### オフライン推論 { #offline-inference }

`LLM` クラスに `enable_sleep_mode=True` を渡すとスリープモードが有効になります。

```python
from vllm import LLM
llm = LLM("Qwen/Qwen3-0.6B", enable_sleep_mode=True)
```

#### Python API { #python-api }

```python
# Sleep level 1
# Put the engine to sleep (level=1: offload weights to CPU RAM, discard KV cache)
llm.sleep(level=1)

# Wake up the engine (restore weights)
llm.wake_up()
```

```python
# Sleep level 2
# Put the engine to sleep (level=2: discard both weights and KV cache)
llm.sleep(level=2)

# Reallocate weights memory only
llm.wake_up(tags=["weights"])

# Load weights in-place
llm.collective_rpc("reload_weights")

# Reallocate KV cache
llm.wake_up(tags=["kv_cache"])
```

#### RLHF の重み更新 { #rlhf-weight-updates }

RLHF の学習中、vLLM では `wake_up()` の tags 引数を使って、モデルの重みまたは KV キャッシュだけを選択的に起こせます。この細かい制御は、モデルの重みを更新するときに特に有用です。重みだけを起こす（例: `llm.wake_up(tags=["weights"])`）ことで、重みの更新が完了するまで KV キャッシュ用のメモリ確保を先送りできます。これにより、重みの同期・更新処理中のピークメモリ使用量を抑えられ、特に大きなモデルで GPU の out-of-memory（OOM）エラーを防ぐのに役立ちます。

どのリソースを復元するかを制御するには `tags=["weights"]` または `tags=["kv_cache"]` を使います。RLHF や重み更新で便利です。**注意**: すべてのコンポーネントが起きるまで、`is_sleeping` は `true` を返します。

```python
# Put engine to deep sleep (level=2)
llm.sleep(level=2)
# ... Get the new weights
# Wake up only weights to avoid OOM
llm.wake_up(tags=["weights"])
# ... Update the weights
# wake up KV cache after weights are updated
llm.wake_up(tags=["kv_cache"])
```

### オンラインサービング { #online-serving }

vLLM サーバーでスリープモードを有効にするには、`VLLM_SERVER_DEV_MODE=1` を付けて起動し、vLLM サーバーに `--enable-sleep-mode` を渡します。

#### 開発モードでのサーバー { #server-in-development-mode }

`VLLM_SERVER_DEV_MODE=1` を指定すると開発用エンドポイントが有効になります。これらのエンドポイントは利用者に公開すべきではありません。

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B \
  --enable-sleep-mode \
  --port 8000
```

レベル 1 でモデルをスリープさせて起こす例は次のとおりです。

```bash
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

レベル 2 でモデルをスリープさせて起こす例は次のとおりです。

```bash
curl -X POST 'http://localhost:8000/sleep?level=2'
# Reallocate weights memory only
curl -X POST 'http://localhost:8000/wake_up?tags=weights'
# Load weights in-place
curl -X POST 'http://localhost:8000/collective_rpc' -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
# Reallocate KV cache
curl -X POST 'http://localhost:8000/wake_up?tags=kv_cache'
```

#### HTTP エンドポイント { #http-endpoints }

- `POST /sleep?level=1` — モデルをスリープさせます（`level=1`）。
- `POST /wake_up` — モデルを起こします。部分的に起こすための任意のクエリパラメータ `tags` に対応しています（例: `?tags=weights`）。
- `POST /collective_rpc` — 集団リモートプロシージャコール（RPC）を実行します。
- `GET /is_sleeping` — モデルがスリープ中かどうかを確認します。

!!! note
    これらのエンドポイントは `VLLM_SERVER_DEV_MODE=1` を指定した場合にのみ利用できます。

## 制限事項 { #limitation }

ROCm では、仮想メモリの確保がチャンク単位で行われます。チャンクサイズは `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE`（単位: MB）で制御できます。既定値は 256MB です。チャンクサイズが大きいほど性能は上がりますが、大きくしすぎると OOM を引き起こします。スリープモードの利用中に OOM が発生した場合は、チャンクサイズを小さくしてみてください。チャンクサイズは 2 のべき乗で指定することを推奨します。
