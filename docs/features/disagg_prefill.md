# プレフィル分離（実験的） { #disaggregated-prefilling-experimental }

このページでは、vLLM のプレフィル分離（disaggregated prefilling）機能を紹介します。

!!! note
    この機能は実験的であり、変更される可能性があります。

## なぜプレフィルを分離するのか { #why-disaggregated-prefilling }

主な理由は 2 つあります。

- **TTFT（time-to-first-token）と ITL（inter-token-latency）を個別に調整できる**。プレフィル分離では、LLM 推論のプレフィル段とデコード段を別々の vLLM インスタンスに配置します。これにより、異なる並列戦略（`tp` や `pp` など）を割り当てられるようになり、ITL に影響を与えずに TTFT を調整したり、TTFT に影響を与えずに ITL を調整したりできます。
- **テールの ITL を制御できる**。プレフィル分離をしない場合、vLLM はあるリクエストのデコード中にプレフィルのジョブを差し込むことがあります。これはテールレイテンシの悪化につながります。プレフィル分離はこの問題を解決し、テールの ITL を制御するのに役立ちます。適切なチャンクサイズを設定したチャンク化プレフィルでも同じ目的は達成できますが、実際には正しいチャンクサイズを見つけるのは困難です。そのため、テールの ITL を制御するにはプレフィル分離のほうがはるかに信頼できる方法です。

!!! note
    プレフィル分離はスループットを向上させるものでは**ありません**。

## 使用例 { #usage-example }

現在、9 種類のコネクタをサポートしています。

- **ExampleConnector**: ExampleConnector を使ったプレフィル分離の例は [examples/disaggregated/example_connector/run.sh](../../examples/disaggregated/example_connector/run.sh) を参照してください。
- **LMCacheConnectorV1**: NIXL を基盤の KV 転送に使う LMCacheConnectorV1 のプレフィル分離の例は [examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh](../../examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh) を参照してください。LMCache は `LMCacheMPConnector` によるマルチプロセス（MP）モードも提供しており、この場合は独立した `lmcache server` が 1 つ以上の vLLM インスタンスで共有される KV キャッシュを保持します。セットアップ方法は [LMCache のサンプル](https://docs.vllm.ai/en/v0.26.0/examples/disaggregated/lmcache/)と [LMCache のドキュメント](https://docs.lmcache.ai)を参照してください。
- **NixlConnector**: 完全な非同期送受信をサポートする NixlConnector のプレフィル分離の例は [tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh](../../tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh) を参照してください。詳しい使い方は [NixlConnector 利用ガイド](nixl_connector_usage.md)を、機能の互換性については [NixlConnector 互換性マトリクス](nixl_connector_compatibility.md)を参照してください。NIXL の転送バックエンドは 1 つまたは複数指定できます。例:

  ```bash
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both", "kv_buffer_device":"cuda", "kv_connector_extra_config":{"backends":["UCX", "GDS"]}}'
  ```

- **MooncakeConnector**: MooncakeConnector を使ったプレフィル分離の例は [examples/disaggregated/mooncake_connector/run_mooncake_connector.sh](../../examples/disaggregated/mooncake_connector/run_mooncake_connector.sh) を参照してください。詳しい使い方は [MooncakeConnector 利用ガイド](mooncake_connector_usage.md)を参照してください。
- **MoRIIOConnector**（ROCm 専用）: 使用例と詳細なドキュメントは [MoRI-IO 利用ガイド](moriio_connector_usage.md)を参照してください。
- **MultiConnector**: KVTransferConfig にすでにある `kv_connector_extra_config: dict[str, Any]` を活用し、使いたいコネクタをキーワード引数の順序付きリストとしてまとめて指定します。例:

  ```bash
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_both"},{"kv_connector":"ExampleConnector","kv_role":"kv_both","kv_connector_extra_config":{"shared_storage_path":"local_storage"}}]}}'
  ```

- **OffloadingConnector**: KV データの CPU メモリへのオフロードを有効にします。CPU 側のブロックサイズ（トークン単位）と確保する CPU メモリの総バイト数を指定できます。

  ```bash
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size": 64, "cpu_bytes_to_use": 1000000000}}'
  ```

  多階層のオフロード（CPU + ファイルシステム階層など）と設定の完全なリファレンスについては、[KV オフロード利用ガイド](kv_offloading_usage.md)を参照してください。

- **FlexKVConnectorV1**: FlexKVConnectorV1 の使用例は [examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py](../../examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py) を参照してください。FlexKV は、超大規模な LLM 推論向けの分散 KV ストアおよび多階層キャッシュ管理システムです。

  ```bash
  --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
  ```

## 開発 { #development }

プレフィル分離は、2 つの vLLM インスタンスを動かすことで実現します。1 つはプレフィル用（プレフィルインスタンスと呼びます）、もう 1 つはデコード用（デコードインスタンスと呼びます）で、コネクタを使ってプレフィルの KV キャッシュと結果をプレフィルインスタンスからデコードインスタンスへ転送します。

プレフィル分離の実装はすべて `vllm/distributed/kv_transfer` 以下にあります。

プレフィル分離における主要な抽象は次のとおりです。

- **Connector**: **kv consumer** が **kv producer** からリクエストのバッチの KV キャッシュを取得できるようにします。
- **LookupBuffer**: KV キャッシュの `insert` と `drop_select` という 2 つの API を提供します。`insert` と `drop_select` のセマンティクスは SQL に似ており、`insert` は KV キャッシュをバッファに挿入し、`drop_select` は指定した条件に一致する KV キャッシュを返してバッファから削除します。
- **Pipe**: テンソル転送のための単方向 FIFO パイプです。`send_tensor` と `recv_tensor` をサポートします。

!!! note
    `insert` は非ブロッキングな操作ですが、`drop_select` はブロッキングな操作です。

上記 3 つの抽象がどのように構成されているかを示す図は次のとおりです。

![Disaggregated prefilling abstractions](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/disagg_prefill/abstraction.jpg)

プレフィル分離のワークフローは次のとおりです。

![Disaggregated prefilling workflow](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/disagg_prefill/overview.jpg)

`buffer` は LookupBuffer の `insert` API に、`drop_select` は LookupBuffer の `drop_select` API に対応します。

現在、vLLM のすべてのプロセスが対応するコネクタを持ちます。具体的には次のとおりです。

- スケジューラコネクタ: スケジューラプロセスと同じプロセスに配置されるコネクタです。KV キャッシュの転送操作をスケジュールします。
- ワーカーコネクタ: ワーカープロセスに配置されるコネクタです。KV キャッシュの転送操作を実行します。

上記 2 つのコネクタがどのように構成されているかを示す図は次のとおりです。

![Disaggregated prefilling high level design](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/disagg_prefill/high_level_design.png)

次の図は、ワーカーコネクタが Attention モジュールと連携して、層ごとの KV キャッシュの保存と読み込みを実現する様子を示しています。

![Disaggregated prefilling workflow](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/disagg_prefill/workflow.png)

## サードパーティによる貢献 { #third-party-contributions }

プレフィル分離はインフラと密接に関係するため、本番レベルのプレフィル分離では vLLM はサードパーティのコネクタに依存しています（vLLM チームはサードパーティコネクタの新しい PR を積極的にレビュー・マージします）。

実装方法として次の 3 つを推奨します。

- **完全にカスタマイズしたコネクタ**: 独自の `Connector` を実装し、サードパーティのライブラリを呼び出して KV キャッシュを送受信します。それ以外にも、カスタマイズしたプレフィルを行うために vLLM のモデル入力を編集するなど、多くのことができます。この方法は最も自由度が高い一方、将来の vLLM のバージョンと非互換になるリスクがあります。
- **データベース的なコネクタ**: 独自の `LookupBuffer` を実装し、SQL のように `insert` と `drop_select` の API をサポートします。
- **分散 P2P コネクタ**: 独自の `Pipe` を実装し、`torch.distributed` のように `send_tensor` と `recv_tensor` の API をサポートします。
