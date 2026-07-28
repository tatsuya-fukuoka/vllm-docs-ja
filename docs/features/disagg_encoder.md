# エンコーダ分離 { #disaggregated-encoder }

**エンコーダ分離（disaggregated encoder）** とは、マルチモーダル LLM のビジョンエンコーダ段を、プレフィル / デコード段とは別のプロセスで実行する方式です。この 2 つの段を独立した vLLM インスタンスとしてデプロイすると、実用上 3 つの利点があります。

1. **独立した細かい粒度のスケーリング**  
2. **TTFT（time-to-first-token）の短縮**  
3. **プロセスをまたいだエンコーダ出力の再利用とキャッシュ**

設計ドキュメント: <https://docs.google.com/document/d/1aed8KtC6XkXtdoV87pWT0a8OJlZ-CpnuLLzmR8l9BAE>

---

## 1  動機 { #1-motivation }

### 1. 独立した細かい粒度のスケーリング { #1-independent-fine-grained-scaling }

* ビジョンエンコーダは軽量ですが、言語モデルは桁違いに大きいものです。  
* エンコーダ群に影響を与えずに言語モデルを並列化できます。  
* エンコーダのノードは独立して追加・削除できます。

### 2. TTFT（time-to-first-token）の短縮 { #2-lower-time-to-first-token-ttft }

* テキストのみのリクエストはビジョンエンコーダを完全にバイパスします。  
* エンコーダの出力は必要な Attention 層にのみ注入されるため、プレフィルのクリティカルパスが短くなります。

### 3. プロセスをまたいだ再利用とキャッシュ { #3-cross-process-reuse-and-caching }

* プロセス内エンコーダでは、再利用の範囲が 1 つのワーカーに閉じてしまいます。  
* リモートの共有キャッシュを使えば、どのワーカーからも既存の埋め込みを取得でき、重複した計算をなくせます。

---

## 2  使用例 { #2-usage-example }

現時点でのリファレンス実装は **ExampleConnector** です。  
次のすぐ実行できるスクリプトがワークフローを示しています。

エンコーダインスタンス 1 + PD インスタンス 1:
`examples/disaggregated/disaggregated_encoder/disagg_1e1pd_example.sh`

エンコーダインスタンス 1 + プレフィルインスタンス 1 + デコードインスタンス 1:
`examples/disaggregated/disaggregated_encoder/disagg_1e1p1d_example.sh`

---

## 3  テストスクリプト { #3-test-script }

`tests/v1/ec_connector` ディレクトリを参照してください。

## 4  開発 { #4-development }

エンコーダ分離は、次の 2 つの部分を動かすことで実現されます。

* **エンコーダインスタンス** – ビジョンエンコードを実行する vLLM インスタンス。  
* **プレフィル / デコード（PD）インスタンス** – 言語側のプレフィルとデコードを実行します。
    * PD は、`disagg_encoder_example.sh` のように 1 つの通常インスタンスにまとめる（E->PD）ことも、`disagg_epd_example.sh` のように分離したインスタンスに分ける（E->P->D）こともできます

コネクタが、エンコーダインスタンスから PD インスタンスへエンコーダキャッシュ（EC）の埋め込みを転送します。  
関連するコードはすべて `vllm/distributed/ec_transfer` 以下にあります。

### 主要な抽象 { #key-abstractions }

* **ECConnector** – エンコーダが生成した EC キャッシュを取得するためのインターフェース。  
    * *スケジューラの役割* – キャッシュの有無を確認し、読み込みをスケジュールします。  
    * *ワーカーの役割* – 埋め込みをメモリに読み込みます。

エンコーダ分離のフローを示す図は次のとおりです。

![Disaggregated Encoder Flow](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/features/disagg_encoder/disagg_encoder_flow.png)

PD 分離の部分では、プレフィルインスタンスは上記のエンコーダ分離のフローとまったく同じ形でキャッシュを受け取ります。プレフィルインスタンスは 1 ステップ（プレフィル → トークン 1 個の出力）を実行し、その後 KV キャッシュをデコードインスタンスに転送して残りの処理を任せます。KV の転送は、PD インスタンスの実行が終わったあとにのみ発生します。

`docs/features/disagg_prefill.md` に、プレフィル分離（v0）の概要が説明されています。

このサンプル構成では、P と D の間の KV 転送を行うために `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` の **NixlConnector** を使い、`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` を参考にしています。
