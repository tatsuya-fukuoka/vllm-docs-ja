# 投機的デコーディング { #speculative-decoding }

このドキュメントでは、中〜低 QPS（1 秒あたりのクエリ数）でメモリ律速なワークロードにおいて、トークン間のレイテンシを削減するために vLLM で[投機的デコーディング](https://arxiv.org/pdf/2302.01318)を使う方法を説明します。

最適化された投機的デコーディングのために自前のドラフトモデルを学習させたい場合は、vLLM とシームレスに学習・統合できる [vllm-project/speculators](speculators.md) を参照してください。

## vLLM の投機手法 { #vllm-speculation-methods }

vLLM はさまざまな投機的デコーディングの手法をサポートしています。EAGLE、MTP、ドラフトモデル、PARD、MLP といったモデルベースの手法は最も高いレイテンシ削減効果をもたらし、n-gram や suffix decoding のような単純な手法は、トラフィックのピーク時に負荷を増やすことなく穏やかな高速化をもたらします。

- [EAGLE](eagle.md)
- [Multi-Token Prediction（MTP）](mtp.md)
- [ドラフトモデル](draft_model.md)
- [並列ドラフトモデル（PARD）](parallel_draft_model.md)
- [多層パーセプトロン](mlp.md)
- [N-Gram](n_gram.md)
- [Suffix Decoding](suffix.md)
- [隠れ状態の抽出](extract_hidden_states.md)
- [カスタムの提案バックエンド（実験的）](#custom-proposer-backend-experimental)
- [動的な投機的デコーディング](dynamic_speculative_decoding.md)

## 手法選択の早見表 { #method-selection-at-a-glance }

手法を選ぶ出発点として、この定性的な表を利用してください。実際の効果は、モデルの系統、トラフィックのパターン、ハードウェア、サンプリングの設定によって変わります。

| 手法 | 低 QPS（レイテンシ重視） | 高 QPS（スループット重視） | 備考 |
| --- | --- | --- | --- |
| EAGLE | 大きい | 中〜大 | 汎用的で強力なモデルベースの手法。 |
| MTP | 大きい | 中〜大 | ターゲットモデルが MTP をネイティブにサポートしている場合に最適。 |
| ドラフトモデル | 大きい | 中程度 | 別途ドラフトモデルが必要。 |
| 並列ドラフトモデル | 大きい | 中〜大 | ドラフトモデルのレイテンシが低い。 |
| MLP speculator | 中〜大 | 中程度 | 互換性のある MLP speculator が利用できる場合に有効。 |
| N-gram | 小〜中 | 中程度 | 軽量で有効化が簡単。 |
| Suffix decoding | 小〜中 | 中程度 | 追加のドラフトモデルが不要。投機の深さが動的。 |
| カスタム提案器 | 場合による | 場合による | 独自の提案器クラスを持ち込む（実験的）。 |
| 動的な投機的デコーディング | 大きい | ベースとなる投機的デコーディング手法より高い | RL や QPS が変動するワークロードで有用。 |

自分の環境で再現可能な測定を行うには、[`examples/features/speculative_decoding/spec_decode_offline.py`](../../../examples/features/speculative_decoding/spec_decode_offline.py) または[ベンチマーク CLI のガイド](../../benchmarking/cli.md)を使ってください。

## カスタムの提案バックエンド（実験的） { #custom-proposer-backend-experimental }

method を `custom_class` に設定し、自作クラスの完全なモジュールパスを指定することで、投機的デコーディングに独自の提案器クラスを組み込めます。カスタムクラスはインスタンス化時に `VllmConfig` を受け取り、`propose` メソッドを実装する必要があります。

**設定例:**

- `speculative_config.method = "custom_class"`
- `speculative_config.model = "your_module.YourCustomProposerClass"`

## `--speculative-config` のスキーマ { #--speculative-config-schema }

CLI では、`--speculative-config` を使って投機的デコーディングの設定を JSON オブジェクトとして渡します。

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "draft_model",
    "model": "<draft-model>",
    "num_speculative_tokens": 5
  }'
```

同じキーは Python からも `LLM(..., speculative_config={...})` で指定できます。以下の表は、この JSON オブジェクトで受け付けられる主要なユーザー向けのキーをまとめたもので、網羅的なスキーマのリファレンスではありません。詳細は、自動生成された[エンジン引数のリファレンス](../../configuration/engine_args.md)と [vllm.config.SpeculativeConfig][] の API ドキュメントを参照してください。

### 共通のキー { #common-keys }

これらのキーは投機的デコーディングの構成全般でよく使われますが、一部は `draft_model`、`mtp`、`eagle3`、`dflash` などのモデルベースの手法にのみ適用されます。

| キー | 型 | 既定値 | 指定できる値 / 意味 |
| --- | --- | --- | --- |
| `method` | `string` | `None` | 投機の手法。よく使われる値は `draft_model`、`ngram`、`suffix`、`mtp`、`eagle3`、`dflash` などです。省略した場合、vLLM は可能な範囲で与えられた設定から手法を推測します。 |
| `model` | `string` | `None` | ドラフトモデル、EAGLE ヘッド、または補助モデルの識別子。`ngram`、`ngram_gpu`、`suffix`、`mtp` では省略できることが多いです。 |
| `num_speculative_tokens` | `integer > 0` | `None` | 1 ステップあたりに提案する投機トークンの数。モデルのメタデータから推測できない手法では必須です。 |
| `draft_tensor_parallel_size` | `integer >= 1` | `None` | ドラフトモデルのテンソル並列サイズ。 |
| `max_model_len` | `integer >= 1` | `None` | ドラフトモデルの最大コンテキスト長。 |
| `parallel_drafting` | `boolean` | `false` | 並列でのドラフトトークン生成を有効にします。EAGLE とドラフトモデルの手法とのみ互換です。 |
| `rejection_sample_method` | `string` | `strict` | `strict`、`probabilistic`、`synthetic` のいずれか。 |
| `synthetic_acceptance_rate` | `float` | `None` | `rejection_sample_method` が `synthetic` のときに目標とする平均受理率。有効な範囲は `[0, 1]`。 |
 | `use_heterogeneous_vocab` | `boolean` | `false` | 語彙が異なるドラフトモデルとターゲットモデルの組み合わせを許可します。初期化時にトークンレベルの共通部分を構築し、ドラフトの logits を共有トークンのみに制約します。`method=draft_model` とのみ互換です。このオプションを有効にした場合、確率的なドラフトサンプリング（`draft_sample_method='probabilistic'`）はまだサポートされていません。 |

!!! note
    Gemma 4 のアシスタントチェックポイントは、汎用のドラフトモデルではなく Gemma 4 の MTP speculator として扱われます。
    [MTP のガイド](mtp.md#gemma-4-assistant-models)に示すとおり、`model` にアシスタントのチェックポイントを指定し
    `"method": "mtp"` を使ってください。

    Gemma 4 のアシスタントチェックポイントに対して起動ログに `SpeculativeConfig(method='draft_model', ...)` と
    表示される場合、インストールされている vLLM のバージョンはその経路の Gemma 4 MTP サポートを含んでいません。
    アシスタントのチェックポイントを汎用のドラフトモデルによる投機的デコーディングに無理やり通すのではなく、
    Gemma 4 MTP のサポートを含むバージョンにアップグレードしてください。

### 手法固有のキー { #method-specific-keys }

#### N-gram { #n-gram }

| キー | 型 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `prompt_lookup_max` | `integer >= 1` | 両方の lookup の上下限を省略した場合は `5`。それ以外で省略した場合は `prompt_lookup_min` と同じ値 | n-gram のウィンドウサイズの最大値。 |
| `prompt_lookup_min` | `integer >= 1` | 両方の lookup の上下限を省略した場合は `5`。それ以外で省略した場合は `prompt_lookup_max` と同じ値 | n-gram のウィンドウサイズの最小値。 |

例:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5
  }'
```

#### Suffix decoding { #suffix-decoding }

| キー | 型 | 既定値 | 意味 |
| --- | --- | --- | --- |
| `suffix_decoding_max_tree_depth` | `integer` | `24` | プレフィックス一致と投機を合わせた木の深さの最大値。 |
| `suffix_decoding_max_cached_requests` | `integer` | `10000` | グローバルな suffix tree にキャッシュするリクエストの最大数。`0` にするとグローバルキャッシュが無効になります。 |
| `suffix_decoding_max_spec_factor` | `float` | `1.0` | 投機の長さを、プレフィックス一致の長さの何倍までに抑えるかを指定します。 |
| `suffix_decoding_min_token_prob` | `float` | `0.1` | トークンを投機するために必要な、推定トークン確率の最小値。 |

例:

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method": "suffix",
    "num_speculative_tokens": 8,
    "suffix_decoding_max_tree_depth": 24,
    "suffix_decoding_max_cached_requests": 10000,
    "suffix_decoding_max_spec_factor": 1.0,
    "suffix_decoding_min_token_prob": 0.1
  }'
```

#### 語彙をまたぐドラフトモデル（TLI） { #cross-vocabulary-draft-models-tli }

  vLLM は既定で、ドラフトモデルとターゲットモデルが同じ語彙を共有していることを要求します。`use_heterogeneous_vocab: true` を設定すると **Token-Level Intersection (TLI)** アルゴリズムが有効になり、異なるトークナイザーを持つ別系統のモデルをドラフトモデルとして使えるようになります。

  初期化時、vLLM はトークン文字列を正規化して共通部分を計算することで、2 つの語彙の間のマッピングを構築します。サンプリングの前にドラフトの logits は共有トークンのみに制約され、サンプリングされたトークン ID は棄却サンプリングの前にターゲット側の語彙へ変換されます。

  ```python
  from vllm import LLM, SamplingParams

  llm = LLM(
      model="Qwen/Qwen3-8B",
      speculative_config={                               
          "method": "draft_model",
          "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
          "num_speculative_tokens": 3,
          "use_heterogeneous_vocab": True,
      },
      gpu_memory_utilization=0.5,
  )
```

### 注意事項 { #notes }

- CLI では `--speculative-config` は JSON オブジェクトを想定しています。YAML の設定ファイルでは、エスケープした JSON 文字列ではなくネストしたマッピングを使ってください。
- `tensor_parallel_size` は `speculative_config` の有効なキーではありません。代わりに `draft_tensor_parallel_size` を使ってください。
- `temperature` や `top_p` などのキーはサンプリングのパラメータであり、`--speculative-config` のフィールドではありません。
- `target_model_config`、`draft_model_config`、`target_parallel_config`、`draft_parallel_config`、`draft_load_config` といった内部フィールドは vLLM が設定するもので、ユーザーが設定することは想定されていません。
- `use_heterogeneous_vocab` は現時点で greedy なドラフトサンプリングのみをサポートします。確率的な受理（temperature > 0 のドラフトサンプリング）はまだサポートされておらず、将来のリリースで追加される予定です。

## 投機的デコーディングのロスレス性の保証 { #lossless-guarantees-of-speculative-decoding }

vLLM において、投機的デコーディングは精度を保ちながら推論効率を高めることを目指しています。このセクションでは、投機的デコーディングのロスレス性の保証を 3 つの観点に分けて説明します。

1. **理論上のロスレス性**
   \- 投機的デコーディングのサンプリングは、ハードウェアの数値精度の限界の範囲で理論上ロスレスです。[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318) で論じられているとおり、浮動小数点の誤差によって出力分布にわずかな差が生じる可能性はあります。

2. **アルゴリズム上のロスレス性**
   \- vLLM の投機的デコーディングの実装は、アルゴリズム的にロスレスであることが検証されています。主な検証テストは次のとおりです。

    > - **棄却サンプラーの収束**: vLLM の棄却サンプラーからのサンプルがターゲットの分布と一致することを確認します。[テストコードを見る](https://github.com/vllm-project/vllm/blob/47b65a550866c7ffbd076ecb74106714838ce7da/tests/samplers/test_rejection_sampler.py#L252)
    > - **greedy サンプリングの一致**: 投機的デコーディングありの greedy サンプリングが、なしの場合と一致することを確認します。これにより、vLLM の投機的デコーディングのフレームワークが、vLLM の forward パスおよび棄却サンプラーと統合された状態でロスレス性を保証していることを検証できます。[tests/spec_decode/e2e](../../../tests/v1/spec_decode) のほぼすべてのテストが、[このアサーションの実装](https://github.com/vllm-project/vllm/blob/b67ae00cdbbe1a58ffc8ff170f0c8d79044a684a/tests/spec_decode/e2e/conftest.py#L291)を使ってこの性質を検証しています。

3. **vLLM の logprob の安定性**
   \- vLLM は現時点で、トークンの対数確率（logprobs）が安定していることを保証していません。そのため、同じリクエストでも実行ごとに異なる出力になることがあります。詳細は [FAQ](../../usage/faq.md) の *vLLM では同じプロンプトでも実行ごとに出力が変わることがありますか?* の項目を参照してください。

vLLM は投機的デコーディングのロスレス性の確保に努めていますが、投機的デコーディングの有無によって生成される出力に差が生じることがあります。その要因は次のとおりです。

- **浮動小数点の精度**: ハードウェアの数値精度の違いにより、出力分布にわずかな差が生じる可能性があります。
- **バッチサイズと数値の安定性**: バッチサイズの変化により logprobs や出力確率が変動することがあります。これは、バッチ処理の非決定的な挙動や数値的な不安定性に起因する可能性があります。

緩和策については、[FAQ](../../usage/faq.md) の *vLLM では同じプロンプトでも実行ごとに出力が変わることがありますか?* の項目を参照してください。

## 既知の機能の非互換性 { #known-feature-incompatibility }

1. `vllm<=0.15.0` の時点では、パイプライン並列と投機的デコーディングは組み合わせられません。
2. `vllm<=0.10.0` では、ドラフトモデルを用いた投機的デコーディングはサポートされていません。

## vLLM のコントリビューター向けリソース { #resources-for-vllm-contributors }

- [[vLLM Office Hours #40] Intro to Speculators](https://www.youtube.com/watch?v=2ISAr_JVGLs)
- [A Hacker's Guide to Speculative Decoding in vLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)
- [What is Lookahead Scheduling in vLLM?](https://docs.google.com/document/d/1Z9TvqzzBPnh5WHcRwjvK2UEeFeq5zMZb5mFE8jR0HCs/edit#heading=h.1fjfb0donq5a)
- [Information on batch expansion](https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit#heading=h.kk7dq05lc6q8)
- [Dynamic speculative decoding](https://github.com/vllm-project/vllm/issues/4565)
