# vLLM V1 { #vllm-v1 }

!!! announcement

    V0 は完全に非推奨となりました。詳細は [RFC #18571](https://github.com/vllm-project/vllm/issues/18571) を参照してください。

    V0 エンジンでは動作するが V1 では動作しないユースケースがある場合は、[GitHub](https://github.com/vllm-project/vllm) または [vLLM Slack](https://inviter.co/vllm-slack) で共有してください。

vLLM V0 は幅広いモデルとハードウェアをサポートしてきましたが、新機能がそれぞれ独立して開発された結果、システムは次第に複雑になっていきました。この複雑さは新機能の統合を難しくし、技術的負債をもたらしたため、より整理された統一的な設計が必要であることが明らかになりました。

vLLM V1 は V0 の成果を土台に、安定して実績のあるコンポーネント（モデル、GPU カーネル、ユーティリティなど）を引き継ぎつつ、
スケジューラ、KV キャッシュマネージャ、ワーカー、サンプラー、API サーバーといったコアシステムを大きく再設計し、
継続的な成長と革新に耐えられる、一貫性があり保守しやすいフレームワークを提供します。

具体的に V1 が目指すのは次の点です。

- **シンプルでモジュール化された、手を入れやすいコードベース**を提供する。
- CPU オーバーヘッドをほぼゼロに抑えた**高い性能**を確保する。
- 主要な最適化を**統一されたアーキテクチャに統合**する。
- 機能や最適化を既定で有効にし、**設定不要**にする。

V1 コアエンジンへの移行により、特に長いコンテキストのシナリオで大幅な性能向上が見られます。性能ベンチマークは今後追加予定です。

詳細は vLLM V1 のブログ記事 [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（2025 年 1 月 27 日公開）を参照してください。

この継続的に更新されるユーザーガイドでは、vLLM V1 で導入された**重要な変更点と制限事項**をまとめています。チームは V1 を既定のエンジンにするため活発に作業を進めており、V1 でサポートされる機能が増えるのに合わせて本ガイドも随時更新されます。

## V0 との違い { #differences-from-v0 }

このセクションでは、V0 と V1 の挙動の違いをいくつか挙げます。

### チャンク化 Prefill { #chunked-prefill }

V0 ではモデルの特性に応じて条件付きで有効化されていましたが、V1 では可能な限り既定で有効になります。

### CUDA グラフ { #cuda-graphs }

V1 の CUDA グラフのキャプチャは、V0 より多くのメモリを消費します。

### Logprobs の意味的な変更 { #semantic-changes-to-logprobs }

#### Logprobs の計算 { #logprobs-calculation }

V1 では既定で、モデルの生の出力から計算された時点で logprobs が返されます（temperature スケーリングやペナルティ調整といった logits の後処理を適用する前の値です）。
そのため、返される logprobs はサンプリング時に実際に使われた調整後の確率を反映していません。

この挙動は `--logprobs-mode` フラグで変更できます。
サポートされるモードは `raw_logprobs`（既定）、`processed_logprobs`、`raw_logits`、`processed_logits` の 4 つです。
raw は bad words などの logit プロセッサを適用する前の値、
processed は temperature や top_k / top_p を含むすべてのプロセッサを適用した後の値を意味します。

#### プレフィックスキャッシュ有効時のプロンプト logprobs { #prompt-logprobs-with-prefix-caching }

V1 はプレフィックスキャッシュを有効にしたままプロンプトの logprobs を返せますが、その logprobs をキャッシュはしません。
プロンプト logprobs を必要とするリクエストでは、エンジンはプレフィックスキャッシュを無視し、logprobs を生成するためにプロンプト全体の Prefill を再計算します。

## 機能のサポート状況 { #feature-support }

各項目の vLLM V1 におけるサポート状況は、次のいずれかです。

- **🟢 利用可能**: 完全に動作し、V0 と同等以上の最適化が行われている。
- **🟡 対応中**: vLLM V1 への導入が予定されており、PR / RFC がオープンになっている。
- **🔴 削除**: vLLM V1 では削除された。強い要望があれば再導入を検討する。

!!! note
    vLLM V1 の統一スケジューラは、プロンプトトークンと出力トークンを同じ方法で扱います。
    単純な辞書（例: `{request_id: num_tokens}`）を用いてリクエストごとに固定のトークン予算を動的に割り当てることで、
    Prefill フェーズと Decode フェーズを厳密に分離せずに、チャンク化 Prefill、プレフィックスキャッシュ、
    投機的デコーディングといった機能を実現しています。

V1 のスケジューラは複数のスケジューリングポリシーをサポートしており、先着順 (FCFS) と
優先度ベースのスケジューリング（割り当てられた優先度順に処理し、同値の場合は FCFS）を
`--scheduling-policy` 引数で選択できます。

### ハードウェア { #hardware }

| ハードウェア  | 状況            |
| --------------| --------------- |
| **NVIDIA**    | <nobr>🟢</nobr> |
| **AMD**       | <nobr>🟢</nobr> |
| **INTEL GPU** | <nobr>🟢</nobr> |
| **TPU**       | <nobr>🟢</nobr> |
| **CPU**       | <nobr>🟢</nobr> |

!!! note

    プラグインを通じて、さらに多くのハードウェアプラットフォームがサポートされる場合があります。例:

    - [vllm-ascend](https://github.com/vllm-project/vllm-ascend)
    - [vllm-spyre](https://github.com/vllm-project/vllm-spyre)
    - [vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
    - [vllm-openvino](https://github.com/vllm-project/vllm-openvino)

    詳細は各リポジトリを確認してください。

### モデル { #models }

| モデルの種類               | 状況                                    |
| -------------------------- | --------------------------------------- |
| **デコーダーのみのモデル** | <nobr>🟢</nobr>                         |
| **エンコーダー・デコーダーモデル** | <nobr>🟢 (Whisper), 🔴 (その他) </nobr> |
| **プーリングモデル**       | <nobr>🟢</nobr>                         |
| **Mamba モデル**           | <nobr>🟢</nobr>                         |
| **マルチモーダルモデル**   | <nobr>🟢</nobr>                         |

V1 でまだサポートされていない、あるいは今後さらに機能追加が予定されているモデルの状況は以下のとおりです。

#### プーリングモデル { #pooling-models }

現在は完全にサポートされており、last プーリングのモデルではプレフィックスキャッシュとチャンク化 Prefill も新たに利用できます。

より多くの種類のプーリングモデルでプレフィックスキャッシュとチャンク化 Prefill を利用できるよう作業を進めています。

#### Mamba モデル { #mamba-models }

標準的な Transformer の Attention ではなく選択的状態空間機構を使うモデルもサポートしています。
Mamba-2 および Mamba-1 の層を使うモデル（`Mamba2ForCausalLM`、`MambaForCausalLM`、`FalconMambaForCausalLM` など）に対応しています。

Mamba-2 / Mamba-1 の層と標準的な Attention 層を組み合わせたハイブリッドモデル（`Zamba2ForCausalLM`、`NemotronHForCausalLM`、`FalconH1ForCausalLM`、`GraniteMoeHybridForCausalLM`、`JambaForCausalLM`、`Plamo2ForCausalLM` など）もサポートしています。

Mamba とは異なる機構を持つハイブリッドモデル（`Lfm2ForCausalLM` など）もサポートしています。

なお、上記いずれのモデルでもプレフィックスキャッシュはまだサポートされていません。

#### エンコーダー・デコーダーモデル { #encoder-decoder-models }

Whisper はネイティブにサポートされています。その他のエンコーダー・デコーダーモデルはプラグインシステム経由でサポートされます。

- **BART**: `BartForConditionalGeneration` は公式の [bart-plugin](https://github.com/vllm-project/bart-plugin) 経由でサポートされます。
- **Florence-2**: `Florence2ForConditionalGeneration` は公式の [bart-plugin](https://github.com/vllm-project/bart-plugin) 経由でサポートされます。

その他のエンコーダー・デコーダーモデル（`MllamaForConditionalGeneration` など）についても、
[プラグインシステム](../design/plugin_system.md)を使って同様のパターンで実装することを推奨します。

### 機能 { #features }

| 機能                                        | 状況                                                                              |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| **プレフィックスキャッシュ**                | <nobr>🟢 利用可能</nobr>                                                          |
| **チャンク化 Prefill**                      | <nobr>🟢 利用可能</nobr>                                                          |
| **LoRA**                                    | <nobr>🟢 利用可能</nobr>                                                          |
| **Logprobs の計算**                         | <nobr>🟢 利用可能</nobr>                                                          |
| **FP8 KV キャッシュ**                       | <nobr>🟢 利用可能</nobr>                                                          |
| **投機的デコーディング**                    | <nobr>🟢 利用可能</nobr>                                                          |
| **プレフィックスキャッシュ有効時のプロンプト logprobs** | <nobr>🟢 利用可能</nobr>                                               |
| **構造化出力の代替バックエンド**            | <nobr>🟢 利用可能</nobr>                                                          |
| **部分 Prefill の並行実行**                 | <nobr>🟡 [対応中](https://github.com/vllm-project/vllm/issues/14003)</nobr>       |
| **best_of**                                 | <nobr>🔴 [削除](https://github.com/vllm-project/vllm/issues/13361)</nobr>         |
| **リクエスト単位の logits プロセッサ**      | <nobr>🔴 [削除](https://github.com/vllm-project/vllm/pull/13360)</nobr>           |
| **GPU <> CPU 間の KV キャッシュスワップ**   | <nobr>🔴 削除</nobr>                                                              |
| **リクエスト単位の構造化出力バックエンド**  | <nobr>🔴 削除</nobr>                                                              |

!!! note

    vLLM V1 の統一スケジューラは、プロンプトトークンと出力トークンを同じ方法で扱います。
    単純な辞書（例: `{request_id: num_tokens}`）を用いてリクエストごとに固定のトークン予算を動的に割り当てることで、
    Prefill フェーズと Decode フェーズを厳密に分離せずに、チャンク化 Prefill、プレフィックスキャッシュ、
    投機的デコーディングといった機能を実現しています。

#### 削除された機能 { #removed-features }

vLLM V1 における大規模なアーキテクチャ刷新の一環として、いくつかの旧来の機能が削除されました。

##### サンプリング関連 { #sampling-features }

- **best_of**: 利用が限られていたため削除されました。詳細は [RFC #13361](https://github.com/vllm-project/vllm/issues/13361) を参照してください。
- **リクエスト単位の logits プロセッサ**: V0 ではリクエストごとに logits を調整する独自の処理関数を渡せましたが、
  vLLM V1 では削除されました。代わりに、起動時に設定する**グローバルな logits プロセッサ**をサポートしています。
  [RFC #17799](https://github.com/vllm-project/vllm/issues/17799) を参照してください。

##### KV キャッシュ関連 { #kv-cache-features }

- **GPU <> CPU 間の KV キャッシュスワップ**: コアアーキテクチャの簡素化により、vLLM V1 ではリクエストのプリエンプションを
処理するために KV キャッシュをスワップする必要がなくなりました。

##### 構造化出力関連 { #structured-output-features }

- **リクエスト単位の構造化出力バックエンド**: 削除されました。現在はフォールバック付きの代替バックエンド（outlines、guidance）がサポートされています。
