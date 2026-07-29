# Model Runner V2 設計ドキュメント { #model-runner-v2-design-document }

## はじめに { #introduction }

vLLM V1 が最初に実装されて以来、私たちはいくつかの根本的な設計上の誤りに気づき、大きな技術的負債を抱えることになりました。当初の設計では想定されていなかった機能が数多く後付けされてきました。同時に、サンプリング手法（Gumbel-max サンプリングなど）、ツール（Triton など）、CUDA の機能（UVA など）についても貴重な知見を得ました。これらの知識をもとに、よりクリーンで効率的、かつモジュール化された設計を目指し、Model Runner V2（MRV2）を第一原理から実装しました。

振り返ってみると、V1 の設計判断の多くは最適とは言えないものでした。MRV2 はまだ機能が完全ではなく、厳密なテストも済んでおらず、未確定の設計判断も残っていますが、V1 に対する大きな改善であると考えています。

このドキュメントでは MRV2 の設計を説明します。

## 1. 永続バッチ { #1-persistent-batch }

V1 における大きな摩擦の原因の 1 つが、永続バッチ（persistent batch）の実装です。

### 背景 { #background }

V1 は、入力準備時の CPU オーバーヘッドを最小化するために永続バッチを導入しました。あるステップでリクエストがスケジュールされると、モデルランナーはモデルに渡すための連続した入力テンソル（ブロックテーブルやリクエストごとの temperature 値など）を構築しなければなりません。これらのテンソルを毎ステップゼロから構築するのは、特にブロックテーブルのような大きなテンソルでは Python 上で非常に低速になりがちです。

永続バッチの最適化は、連続するステップのリクエストバッチがほぼ同一であるという事実を利用します。1 ステップあたりに参加・完了するリクエストは（あったとしても）ごくわずかです。永続的な状態テンソルを保持し、入力をゼロから作り直すのではなく差分を適用することで、CPU のオーバーヘッドを大幅に削減できます。

### V1 のアプローチの問題点 { #problems-with-v1s-approach }

効率的ではあるものの、V1 の永続バッチ設計は、永続状態と入力テンソルを結合してしまったために不必要な複雑さを生みました。V1 は永続状態テンソルをそのままモデルとサンプラーの入力として使うため、レイアウトと順序に厳しい制約が課されます。リクエストが参加・完了する際には、単純な行の挿入 / 削除ではなく、テンソル全体の複雑な並べ替えが必要になることがよくあります。

また V1 では、リクエストがまだアクティブなうちに永続テンソルの行が上書きされうるため、リクエスト状態の冗長なバックアップである `CachedRequestState` を保持する必要がありました。

その結果、非同期スケジューリングのもとではさらに難しくなる、複雑な管理処理が生まれています。

![Persistent Batch in V1](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/model_runner_v2/persistent_batch_v1.png)

### MRV2 の解決策 { #mrv2s-solution }

MRV2 は、永続状態テンソルとステップごとの入力テンソルを分離します。そのステップのリクエスト順序（通常は Attention バックエンドが決定します）が与えられると、MRV2 は永続状態から入力テンソルを gather します。

1. `max_num_reqs` 行（多くのプラットフォームで既定 1024）の固定サイズのテンソルを事前に確保する。
2. 各リクエストに、アクティブな間（完了またはプリエンプションまで）ずっと使う固定の行を割り当てる。
3. プリエンプションは完了として扱う。再開時には、新しい状態としてリクエストのデータを追加し直す。

これにより `CachedRequestState` が不要になり、管理処理が簡素化されます。大きな状態テンソルはほとんどが GPU メモリ上に置かれるため、gather は GPU 上で並列に、低いオーバーヘッドで実行されます。

![Persistent Batch in MRV2](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/model_runner_v2/persistent_batch_mrv2.png)

## 2. 非同期を前提とした設計 { #2-async-first }

vLLM は現在、非同期スケジューリングに大きく依存しています。GPU がステップ `N` を実行している間に、スケジューラとワーカーがステップ `N+1` の入力を準備し、CPU と GPU の処理を重ね合わせて利用率を最大化します。

V1 はもともと非同期スケジューリングを念頭に設計されておらず、対応には後付けの挙動や場当たり的な工夫が必要でした。一方 MRV2 では、モデル実行の中核ループを CPU の同期点を持たない CUDA ストリームとみなします。CPU 側のエントリポイントは、そのストリームに処理をキューイングします。

![Async execution timeline](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/model_runner_v2/async_sched.png)

## 3. 非同期バリアの排除 { #3-removing-async-barrier }

非同期実行の重要な要件は、CPU の処理がブロッキングしないことです。明示的な同期（`torch.accelerator.synchronize` など）も、暗黙的な同期（pinned でないメモリからの `.to("cuda")` など）も避けなければなりません。

しかし非同期実行では、CPU と GPU が同じメモリに同時に触れることで競合状態が生じる可能性があります。

例（安全でないコード）:

```python
class ModelRunner:
    def __init__(self, ...):
        # Pinned buffer
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=True
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data
        states = self.states.to("cuda", non_blocking=True)
```

GPU が非同期コピーで `self.states` をまだ読んでいる間に、CPU がそれを書き換えてしまう可能性があります。

V1 はこれに対して、クリティカルセクションを非同期バリアで囲むことで対処しています。競合は避けられますが、次の欠点があります。

1. 保護すべきバッファを見落としやすい（バグを生みやすい）。
2. 構成が硬直的（CPU の処理をすべてバリアの内側に収める必要がある）。
3. 同期のためにオーバーラップが減る可能性がある。

![Race condition with shared CPU buffer](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/model_runner_v2/async_race_condition.png)

### MRV2 の解決策: 競合そのものをなくす { #mrv2s-solution-eliminate-the-race }

MRV2 は、永続的な CPU 側の状態と、コピー対象のテンソルを分離します。

```python
class ModelRunner:
    def __init__(self, ...):
        # Not pinned
        self.states = torch.zeros(
            max_num_reqs, dtype=torch.int32, device="cpu", pin_memory=False
        )

    def execute_step(self, ...):
        self.states[req_idx] = new_req.data
        tmp_states = self.states.pin_memory()
        states = tmp_states.to("cuda", non_blocking=True)
```

これにより、CPU は `self.states` に書き込み、GPU は `tmp_states` から読むことになり、明示的な同期なしに競合をなくせます。

![No race with temporary pinned copy](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/model_runner_v2/async_no_race_condition.png)

## 4. StagedWriteTensor { #4-stagedwritetensor }

ブロックテーブルのような大きなテンソルについて、MRV2 は `StagedWriteTensor` を使うことで、毎ステップの CPU から GPU への全体コピーを避けます。

1. ベースとなるテンソルを GPU 上に保持する。
2. 差分を CPU 側にステージングする。
3. 差分を連続したバッファにパックする。
4. パックした差分を GPU にコピーする。
5. 差分を適用するカーネルを 1 回だけ起動する。

使用例:

```python
# Initialize state on GPU
state = StagedWriteTensor(size=(1024, 1000), dtype=torch.int32, device="cuda")

# Write [3, 1, 2] into row 2, starting at index 3
state.stage_write(row=2, start=3, value=[3, 1, 2])

# Write [-1, -2, -5] into row 0, starting at index 1
state.stage_write(row=0, start=1, value=[-1, -2, -5])

# Apply staged changes
state.apply_write()
```

これにより、CPU と GPU の同期なしで、最小限のカーネル起動で不揃いな（ragged な）更新に対応できます。ブロックテーブルや、`num_computed_tokens` のように CPU と GPU の双方から書き込まれる状態に特に有用です。

## 5. GPU ネイティブな入力メタデータ準備と出力処理 { #5-gpu-native-input-metadata-preparation-and-output-processing }

MRV2 は、`input_ids`、`positions`、`query_start_loc`、`seq_lens` といった入力の準備に Triton カーネルを使います。

利点:

1. 非同期性が向上する: CPU がまだ知り得ない値（投機的デコーディングの場合など）を GPU 側で導出できます。
2. CPU のオーバーヘッドが下がる: 入力準備は GPU 上では非常に安価で、Python のボトルネックを避けられます。

### Universal Virtual Addressing（UVA） { #universal-virtual-addressing-uva }

MRV2 は一部の経路で UVA を使い、CPU 上にある大きなテンソル（`prefill_token_ids` など）を GPU メモリに複製することなく、GPU カーネルから直接アクセスできるようにしています。

## 6. Triton ネイティブなサンプラー { #6-triton-native-sampler }

MRV2 は、数値計算とメモリの制御・最適化を高めるため、サンプリングの大部分を Triton で再実装しています。

### Gumbel サンプリングカーネル { #gumbel-sampling-kernel }

MRV2 は、softmax を明示的に実体化することを避け、シード入力からステートレスなカーネル内 RNG を用いる Triton の Gumbel サンプリングカーネルを導入しています。

### 効率的な top-k logprobs { #efficient-top-k-logprobs }

V1 は top-k を求める前に語彙全体の logprobs を実体化します。MRV2 はまず logits から top-k のトークンを特定し、選ばれたトークンについてのみ logprobs を計算します。これにより GPU メモリのピーク使用量が削減されます。

### メモリ効率の良いプロンプト logprobs { #memory-efficient-prompt-logprobs }

MRV2 は、1 つのプロンプトの内部での分割も含む、より細かい粒度のチャンク化をサポートし、長いプロンプトでのメモリのスパイクを避けます。

### 投機的デコーディングとの相性の向上 { #better-compatibility-with-speculative-decoding }

MRV2 は、リクエストごとのサンプリング状態を logit ごとの形状に合わせて展開する代わりに、カーネル内で間接参照（`idx_mapping`）を使い、各 logits ベクトルを正しいリクエスト状態に対応づけます。これにより、複雑なサンプリングパラメータや logits processor のサポートが簡素化されます。

## 7. モジュール性 { #7-modularity }

MRV2 はモジュール性を重視しています。V1 の巨大で絡み合った `gpu_model_runner.py` と比べ、MRV2 は機能ごとのロジックを専用のファイル（`mrope_utils.py`、`penalties.py` など多数）に分割しています。

また、モデルの入力を `InputBatch` クラスにまとめ、モデルランナーの属性への直接的な結合を減らしています。

## 8. `dummy_run` を濫用しない { #8-no-abuse-of-dummy_run }

V1 では `dummy_run` が多くの役割を抱え込んでいました。

- 初期のメモリプロファイリングと `torch.compile`
- CUDA graph のキャプチャ
- ウォームアップ
- EP + DP のための空の DP forward パス

MRV2 ではこれを簡素化しています。

1. `execute_model` が、状態に影響を与えないダミー実行をサポートする。
2. `dummy_run` は、プロファイリング・ウォームアップ・空の DP forward パスを `execute_model` に委譲する。
3. CUDA graph のキャプチャは、専用の別経路を使う。

これにより複雑さが減り、`execute_model` と `dummy_run` の挙動の乖離に起因するバグがなくなります。

## 9. 明示的な CUDA graph 管理 { #9-explicit-cuda-graph-management }

V1 の CUDA graph の扱いは暗黙的で、理解しにくいものでした。MRV2 は `CUDAGraphManager` を使い、標準の PyTorch API を通じて full CUDA graph を明示的にキャプチャ・起動します。

これにより、グラフのライフサイクルと実行モードの判断が理解しやすくなり、拡張も容易になります。たとえば MRV2 では、複数のドラフトモデルの forward パスを 1 つの CUDA graph にキャプチャできます。

## 開発の考え方 { #development-philosophy }

MRV2 への変更には、より高いコード品質の水準が求められます。V1 との機能差を埋めていく際には、V1 の挙動を素早く移植するのではなく、MRV2 の設計文脈のもとで第一原理から機能を再検討すべきです。

重要な要件は、たとえ事前の設計検討により多くの反復が必要になったとしても、モジュール性ときれいな抽象の境界を保つことです。
