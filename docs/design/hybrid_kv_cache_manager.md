# ハイブリッド KV キャッシュマネージャ { #hybrid-kv-cache-manager }

!!! warning
    このドキュメントはコミット [458e74](https://github.com/vllm-project/vllm/commit/458e74eb907f96069e6d8a4f3c9f457001fef2ea) をもとに書かれています。この機能はまだ初期段階にあり、内容は変わる可能性があります。

## ハイブリッドモデルとは { #what-is-a-hybrid-model }

近年の「ハイブリッド」LLM の多くは、1 つのモデルの中に複数の attention 種別を組み合わせています。例:

1. スライディングウィンドウ attention（sw）+ フル attention（full）: gpt-oss、Gemma 2/3、Ministral、cohere など
2. Mamba + full: Bamba、Jamba、Minimax など
3. ローカルなチャンク attention + full: Llama4

これらのモデルを効率よくサービングするには、vLLM の [`KVCacheManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_manager/#vllm.v1.core.kv_cache_manager.KVCacheManager) が次を満たす必要があります。

1. 層の種別ごとに異なるスロットを割り当てる。例:
    - フル attention 層: **すべての**トークン分のスロットを確保する。
    - スライディングウィンドウ層: 直近の **`sliding_window_size`** トークン分のスロットのみ確保する。
2. 層ごとに異なるプレフィックスキャッシュのルールをサポートする。例:
    - フル attention: キャッシュヒットするプレフィックスには、**すべての**トークンが KV キャッシュに残っている必要がある。
    - スライディングウィンドウ: キャッシュヒットするプレフィックスには、末尾の **`sliding_window_size`** トークンが KV キャッシュに残っていればよい。

## 用語の定義 { #definitions }

1. **kv hidden size**: 1 層分について、1 トークンの KV キャッシュを保存するのに必要なバイト数。
2. **block**: KV キャッシュ用に確保したメモリは、同じ *page size*（下記参照）を持つ複数の *block* に分割されます。
3. **block size**: 1 つの block に含まれるトークン数。
4. **page size**: 1 つの block の物理メモリサイズ。次のように定義されます。

    $$
    \text{num_layers} \times \text{block_size} \times \text{kv_hidden_size}
    $$

    `num_layers` はモデル全体の層数を意味するわけではありません。実際の値は、このドキュメント内の文脈によって異なります。

    !!! note
        これはコード中の `KVCacheSpec.page_size_bytes` とは異なります。そちらは次のように定義されています。

        $$
        \text{block_size} \times \text{kv_hidden_size}
        $$

## メモリの割り当て { #allocation }

### 全体の考え方 { #high-level-idea }

すべての層の種別で単一のメモリプールを使います。このメモリプールは、同じ page size を持つ複数の block に分割されます。[`KVCacheManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_manager/#vllm.v1.core.kv_cache_manager.KVCacheManager) は、attention の種別に応じて層ごとに異なる数の block を割り当てます。

中心的な課題は、すべての層の種別で同じ **page size** を使うようにすることです。フル attention のみのモデルでは page size は単純で、次のように定義されます。

$$
\text{page_size} = \text{block_size} \times \text{num_hidden_layers} \times \text{kv_hidden_size}
$$

ところがハイブリッドモデルでは `num_hidden_layers` が attention の種別ごとに異なるため、そのままでは page size が食い違ってしまいます。以下のケースでは、これをどう統一するかを説明します。

### ケース 1: おもちゃのモデル { #case-1-toy-model }

まずは簡単な例から始めます。フル attention 層が 1 つ、スライディングウィンドウ attention 層が 3 つあるモデルを考えます。すべての層の `kv_hidden_size` は同じとします。

各 block が 1 層分の `block_size` トークンを保持するようにすると、次のようになります。

$$
\text{page_size} = \text{kv_hidden_size} \times \text{block_size}
$$

[`KVCacheManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_manager/#vllm.v1.core.kv_cache_manager.KVCacheManager) は、各層に異なる数の block を割り当てます。

このケースはあくまで簡略化した例です。実際のモデルについては、以降のケースを参照してください。

### ケース 2: `kv_hidden_size` が同じで規則的なパターンがある場合 { #case-2-same-kv_hidden_size-and-a-regular-pattern }

モデルの層数が増える場合を考えます。たとえば `kv_hidden_size` が同じスライディングウィンドウ attention 層 20 個とフル attention 層 10 個からなるモデルです。層ごとにアロケータを呼ぶ（30 回の呼び出し）こと自体は可能ですが、非効率になります。そこで、同じ数の block を必要とする層の割り当てをまとめることで、呼び出し回数を減らします。

このグループ化が成り立つのは、通常、種別ごとの層数のあいだにきれいな比があるからです。例:

- Gemma-2: 1 sw : 1 full
- Llama 4: 3 local : 1 full

この例のモデルは 2 sw : 1 full と見なせます。モデルに 2 つの sw と 1 つの full があるものとして block を割り当て、その結果を 10 回繰り返すことで 30 層分の `block_ids` を生成できます。page size は次のようになります。

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

`block_size` を 16、スライディングウィンドウのサイズを 32、リクエスト長を 112 とすると、上記の例のモデルでは 11 個の block を割り当てる必要があります（full に 0〜6、sw グループ 1 に 7〜8、sw グループ 2 に 9〜10）。

![割り当て結果](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/hybrid_kv_cache_manager/basic_grouping_example.png)

ここで「/」は block が不要であることを示します（スライディングウィンドウ層は、前方のトークンにスロットを必要としません）。

以下に形式的な定義を示します。層は複数の *KV キャッシュグループ* に分割され、次の 2 つの性質を満たします。

1. **各グループ内で attention の種別が同一**: 各グループには同じ attention 種別の層だけが含まれるため、あるリクエストに対して必要な block 数も同じになります。これにより、同じグループの層がメモリを無駄にせず同じ block ID を共有できます。
2. **グループ間で page size が同一**: メモリプールが単一の page size しか持たないためです。

例のモデルは 3 つの KV キャッシュグループに分割されます。

- グループ 0: フル attention 層 10 個（full.0 〜 full.9）
- グループ 1: スライディングウィンドウ attention 層 10 個（sw.0 〜 sw.9）
- グループ 2: スライディングウィンドウ attention 層 10 個（sw.10 〜 sw.19）

これが規則 1 を満たすのは明らかです。規則 2 についても、3 つのグループはいずれも

$$
10 \times \text{kv_hidden_size} \times \text{block_size}
$$

を page size として持ちます。

### ケース 3: `kv_hidden_size` が同じで規則的なパターンがない場合 { #case-3-same-kv_hidden_size-and-no-regular-pattern }

残念ながら、すべてのモデルがこのようなきれいな比を持つわけではなく、ケース 2 の方法では小さなグループが多くなりすぎます。たとえば Gemma-3-27b はスライディングウィンドウ attention 層 52 個とフル attention 層 10 個を持ちます。ケース 2 の制約では、それぞれ 2 層からなるスライディングウィンドウのグループ 26 個とフル attention のグループ 5 個になり、割り当ては依然として非効率です。KV キャッシュグループの数を減らすため、すべての attention 種別のうち最も少ない層数を基準に層をグループ化します。たとえば Gemma-3-27b では min(52, 10)=10 層が 1 グループになります。この場合のグループ化結果は次のとおりです。

- グループ 0: フル attention 層 10 個（full.0 〜 full.9）
- グループ 1: スライディングウィンドウ attention 層 10 個（sw.0 〜 sw.9）
- グループ 2: スライディングウィンドウ attention 層 10 個（sw.10 〜 sw.19）
- ...
- グループ 6: スライディングウィンドウ attention 層 10 個（sw.40 〜 sw.49）
- グループ 7: スライディングウィンドウ attention 層 2 個（sw.50 〜 sw.51）とパディング用の 8 層

新しいモデルが登場して、このヒューリスティックが良くない結果を招く場合（たとえば 20 full + 30 sw では、グループサイズは 20 ではなく 10 にすべきです）には、アルゴリズムを更新します。

このケースは Gemma-3 系のモデル、およびケース 2 に該当するモデルにフル attention 層を 1 つ追加する eagle 投機的デコーディングを組み合わせた場合に発生します。この解決策にはある程度のメモリの無駄があり、完璧ではありません。パディングのオーバーヘッドが許容できないケースがあれば報告してください。アルゴリズムの改善に役立てます。

### ケース 4: `kv_hidden_size` が異なる場合（主にハイブリッド mamba モデル） { #case-4-different-kv_hidden_size-mainly-hybrid-mamba-models }

一部のアーキテクチャ（Bamba、Jamba、Minimax など）では、標準的な attention 層と Mamba 層が交互に配置されます。Mamba 層の 1 トークンあたりの状態サイズは、attention 層の `kv_hidden_size` よりずっと大きくなることがあります。すべてのグループで単一の page size しかサポートしないため、これら異なる hidden size を揃える必要があります。

現在のアルゴリズムは次のとおりです。

1. attention 層の `block_size` を、次を満たすまで大きくします。
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \ge \text{state_size}_{\text{mamba}}
    $$
2. 各層の mamba の状態を次のサイズまでパディングします。
    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}}
    $$
3. ケース 3 のグループ化戦略を適用します。

!!! note
    この方法では attention 層の `block_size` が 400 を超えることがあり、大きすぎます。別のパディング戦略として、次を満たすまで `block_size` を大きくする方法があります。

    $$
    \text{block_size} \times \text{kv_hidden_size}_{\text{att}} \times \text{num_attn_layers} \ge \text{state_size}_{\text{mamba}}
    $$

    このパディング戦略はまだ作業中です。

### ケース 5: KV 共有 { #case-5-kv-sharing }

KV 共有（KV sharing）とは、ある層が別の層の KV キャッシュを利用することを指します（gemma-3n など）。
これらのモデルでは、[`KVCacheManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_manager/#vllm.v1.core.kv_cache_manager.KVCacheManager) は KV 共有を行うすべての層を無視し、KV キャッシュが必要な層に対してのみ割り当てを行います。そして、割り当て結果を KV 共有の層に適用するためのパッチが model runner 側に入っています。

## プレフィックスキャッシュ { #prefix-caching }

説明を簡単にするため、この節では `block_size=1` と仮定します。

### 全体の考え方 { #high-level-idea_1 }

block プールは、`tuple(block_hash, group_id) -> block` のような辞書を使って埋まった block をキャッシュします。つまり、異なるグループの同じトークンは、それぞれ独立にキャッシュ・追い出しされます。

新しいリクエストが来ると、各グループについてキャッシュヒットするプレフィックスを調べ、それらの共通部分をリクエストのキャッシュ済みプレフィックスとして返します。1 グループのキャッシュヒットを調べる方法と共通部分の求め方の詳細は、以下を参照してください。

### ケース 0: フル attention のみのモデル { #case-0-full-attention-only-models }

フル attention 層では、リクエスト内のすべてのトークンに block が割り当てられます。基礎となる設計の詳細は[プレフィックスキャッシュ](prefix_caching.md)を参照してください。

リクエストの最長キャッシュヒットプレフィックスを求めるには、左（最初の block）から右（最後の block）へ順に block がキャッシュされているかを調べ、キャッシュミスしたところで終了します。たとえば次の例では、最初の 7 トークン（0〜6）がキャッシュヒットプレフィックスとして返されます（青い block がキャッシュ済み）。

![フル attention のプレフィックスキャッシュ](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/hybrid_kv_cache_manager/full_attn.png)

### ケース 1: スライディングウィンドウ attention のみのモデル { #case-1-sliding-window-attention-only-models }

スライディングウィンドウ attention 層では、素朴なメモリ割り当ての実装として、`sliding_window_size` 個の block を確保してラウンドロビンで埋めていく方法が考えられます。しかしこの素朴な実装はプレフィックスキャッシュと両立しないため、この設計は採用していません。vLLM では、トークンごとに異なる block を割り当て、スライディングウィンドウの外に出た block を解放します。

新しいリクエストでは、キャッシュヒットするプレフィックスに必要なのは、末尾の `sliding_window_size - 1` トークンがキャッシュされていることだけです。
`sliding_window_size = 4`、`block_size = 1` で、リクエストが 15 トークンのプロンプトである場合を考えます（青い block がキャッシュ済み）。

![スライディングウィンドウ attention のプレフィックスキャッシュ](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/hybrid_kv_cache_manager/sw_attn.png)

キャッシュヒットするプレフィックスの候補は 3 つあります。

- キャッシュヒット長 5、[2, 3, 4] → [5, 6, …, 14] のプレフィルを計算
- キャッシュヒット長 6、[3, 4, 5] → [6, 7, …, 14] のプレフィルを計算
- キャッシュヒット長 14、[11, 12, 13] → [14] のプレフィルを計算（最も効率的）

キャッシュヒットは右から左へ調べ、一致が見つかった時点で早期終了できます。これはフル attention とは逆です（フル attention では左から右へ調べ、一致しなくなった時点で早期終了します）。フル attention と比べたときの潜在的な欠点は、一致がまったくない場合にトークン列全体を走査することになる点で、これは珍しくない状況です。無視できないオーバーヘッドを生む可能性がありますが、以下で述べるように full + swa の構成では問題になりません。

### ケース 2: スライディングウィンドウ attention + フル attention のモデル { #case-2-sliding-window-attention-full-attention-models }

第 1 の課題は、キャッシュヒットするプレフィックスをどう求めるかです。グローバル attention 層とスライディングウィンドウ attention 層のキャッシュヒットを、次の手順で「交差」させる必要があります。

1. フル attention について最長のキャッシュヒットを求める（左から右へ走査）。
2. その長さの範囲内で、スライディングウィンドウ attention の最長のキャッシュヒットを求める。実装としては、フル attention のキャッシュヒット長を起点に右から左へキャッシュヒットを調べます。

この手順により、得られたスライディングウィンドウ attention 層のキャッシュヒットが、フル attention 層のキャッシュヒットでもあることが保証されます。各グループのあり得るプレフィックスをすべて求めて共通部分を取るよりも効率的です。キャッシュヒットがない場合に早期終了できるためです。

このアルゴリズムは、attention の種別がちょうど 2 つ、すなわち「フル attention + X」であるモデルに適用できます。X は、スライディングウィンドウ、llama 4 のローカル attention、mamba など任意の効率的な attention アルゴリズムです。フル attention 層を持たないモデルや、3 種類以上の attention を持つモデルはサポートしません。このドキュメントの執筆時点では、ほとんどのハイブリッドモデルにとってこれで十分です。

第 2 の課題はキャッシュの追い出しポリシーです。現時点では、すべての KV キャッシュグループに対して 1 つの LRU キューを使います。block は解放されたとき（リクエストが完了した、あるいは block がスライディングウィンドウの外に出た）に LRU キューへ追加されます。

### ケース 3: mamba モデル { #case-3-mamba-models }

mamba モデルのプレフィックスキャッシュ対応は作業中です。実装されれば、mamba 層 + フル attention 層のモデルも、ケース 2 の「フル attention + X」アルゴリズムでサポートできるようになります。

## 実装 { #implementation }

### 概要 { #overview }

![ハイブリッド KV キャッシュマネージャの概要](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/hybrid_kv_cache_manager/overview.png)

`KVCacheManager` は 3 つの層に分かれています。

- **[`KVCacheManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_manager/#vllm.v1.core.kv_cache_manager.KVCacheManager)**: スケジューラと KV キャッシュ管理システムのあいだのインターフェース。
- **[`KVCacheCoordinator`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_coordinator/#vllm.v1.core.kv_cache_coordinator.KVCacheCoordinator)**: グループごとの SingleTypeKVCacheManager を調停し、リクエストの割り当て結果を生成します。モデルの構成に応じて、次のいずれかの coordinator が選ばれます。
    - **[`KVCacheCoordinatorNoPrefixCache`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_coordinator/#vllm.v1.core.kv_cache_coordinator.KVCacheCoordinatorNoPrefixCache)**: プレフィックスキャッシュが無効な場合に使われます。
    - **[`UnitaryKVCacheCoordinator`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_coordinator/#vllm.v1.core.kv_cache_coordinator.UnitaryKVCacheCoordinator)**: KV キャッシュグループが 1 つだけの場合に使われます。共通部分を取る必要がないため、プレフィックスキャッシュのロジックが簡略化されます。
    - **[`HybridKVCacheCoordinator`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/kv_cache_coordinator/#vllm.v1.core.kv_cache_coordinator.HybridKVCacheCoordinator)**: ちょうど 2 つの KV キャッシュグループを扱います（フル attention のグループ 1 つと、それ以外の効率的な attention のグループ 1 つを含む必要があります）。それ以外のケースは未実装です。プレフィックスキャッシュを無効にすれば KVCacheCoordinatorNoPrefixCache を使えます。
- **[`SingleTypeKVCacheManager`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/core/single_type_kv_cache_manager/#vllm.v1.core.single_type_kv_cache_manager.SingleTypeKVCacheManager)**: 各インスタンスが 1 つの KV キャッシュグループについて割り当てとプレフィックスキャッシュを管理し、attention 種別ごとのロジック（フル attention、スライディングウィンドウ、Mamba など）を実装します。

上図の青い枠は、フル attention 層 10 個とスライディングウィンドウ attention 層 20 個の場合を示しています。したがって次のようになります。

- `HybridKVCacheCoordinator` を使用
- 3 つの `KVCacheGroup` に対して `FullAttentionManager` を 1 つ、`SlidingWindowManager` を 2 つ使用

### メモリレイアウト { #memory-layout }

n 個の `KVCacheGroup`（各グループが m 層）を持つモデルでは、m 個のバッファを確保します。各バッファは、各グループから 1 層ずつ、計 n 層で共有されます。

次の図は、フル attention 層 10 個（full.0 〜 full.9）とスライディングウィンドウ attention 層 20 個（sw.0 〜 sw.19）を持つモデルの例です。「メモリの割り当て」節の「ケース 2」に従って、3 つのグループに分割されています。

- グループ 0: フル attention 層 10 個（full.0 〜 full.9）
- グループ 1: スライディングウィンドウ attention 層 10 個（sw.0 〜 sw.9）
- グループ 2: スライディングウィンドウ attention 層 10 個（sw.10 〜 sw.19）

そしてあるリクエストに対して、`block_id` 0〜6 をグループ 0、7〜8 をグループ 1、9〜10 をグループ 2 に割り当て、計 11 個の block を確保します。

この例では、物理メモリは 10 個のバッファ（`KVCacheTensor` 0 〜 `KVCacheTensor` 9）に分割されます。各バッファは 3 つの層で共有され（たとえば `KVCacheTensor` 0 は、グループ 0 の full.0、グループ 1 の sw.0、グループ 2 の sw.10 で共有されます）、`block_size * kv_hidden_size` のサイズの区画に分割されます。これら 3 つの attention 層の KV キャッシュは、割り当てられた `block_ids` にもとづいてバッファの異なる区画に保存されます。

![メモリレイアウトの例](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/hybrid_kv_cache_manager/memory_layout.png)

!!! note
    論理的な 1 つの「block」は、物理メモリ上の 10 個のバッファ内の 10 個の区画に対応づけられます。
