# 自動プレフィックスキャッシュ { #automatic-prefix-caching }

KV キャッシュのブロックをプレフィックス単位でキャッシュすることは、プロンプトの重複計算を避けるために LLM 推論で広く使われる最適化です。考え方はシンプルで、処理済みリクエストの KV キャッシュブロックをキャッシュしておき、以前のリクエストと同じプレフィックスを持つ新しいリクエストが来たときにそれらのブロックを再利用します。プレフィックスキャッシュはほぼコストなしで得られる恩恵であり、モデルの出力も変えないため、多くの公開エンドポイント（OpenAI、Anthropic など）やほとんどのオープンソース LLM 推論フレームワーク（SGLang など）で広く採用されています。

プレフィックスキャッシュの実装方法は複数ありますが、vLLM はハッシュベースのアプローチを採用しています。具体的には、各 KV キャッシュブロックを、そのブロック内のトークンと、そのブロックより前のプレフィックスのトークンからハッシュ化します。

```text
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

上の例では、最初のブロックの KV キャッシュはトークン列「A gentle breeze stirred」で一意に識別できます。3 番目のブロックは、ブロック内のトークン「laughed in the distance」と、プレフィックスのトークン「A gentle breeze stirred the leaves as children」を合わせることで一意に識別できます。したがって、`hash(tuple[components])` という形でブロックのハッシュを構成できます。components は次のとおりです。

* 親のハッシュ値: 親のハッシュブロックのハッシュ値。
* ブロックのトークン: このブロック内のトークンのタプル。正確なトークンを含めるのは、ハッシュ値の衝突の可能性を減らすためです。
* 追加のハッシュ: このブロックを一意にするために必要なその他の値。LoRA の ID、マルチモーダル入力のハッシュ（後述の例を参照）、マルチテナント環境でキャッシュを分離するためのキャッシュソルトなど。

!!! note "注 1"
    キャッシュするのは満杯のブロックのみです。

!!! note "注 2"
    以前のバージョンでは、ハッシュキーが衝突しないことは保証されていませんでした。v0.11 以降、既定のハッシュアルゴリズムは `sha256` になり、衝突のリスクに対処しています。

    `vllm serve` では、`--prefix-caching-hash-algo` でハッシュアルゴリズムを制御できます。
    - `sha256`（既定）: シリアライズに Python の `pickle` を使います。ハッシュは Python や vLLM のバージョンが異なると再現しない場合があります。
    - `sha256_cbor`: シリアライズに `cbor2` を使い、再現可能で言語間の互換性があるハッシュを提供します。環境をまたいで決定的なキャッシュを行いたい場合に推奨されます。
    - `xxhash`: Pickle によるシリアライズと xxHash（128 ビット）を組み合わせ、より高速な非暗号学的ハッシュを行います。オプションの `xxhash` パッケージが必要です。重要: 暗号学的に安全とはみなされないハッシュアルゴリズムを使うと、理論上はハッシュ衝突のリスクが高まり、未定義動作や、マルチテナント環境でのプライベート情報の漏えいにつながる可能性があります。衝突の可能性は依然として非常に低いとはいえ、有効にする前に、性能上の利点とセキュリティリスクの許容度を比較検討することが重要です。
    - `xxhash_cbor`: 正準な CBOR シリアライズと xxHash を組み合わせ、再現可能なハッシュを行います。オプションの `xxhash` パッケージが必要です。    

**マルチモーダル入力でのハッシュの例**  
ここでは、マルチモーダル入力（画像など）でプレフィックスキャッシュがどう動作するかを説明します。次のメッセージを持つリクエストがあるとします。

```text
messages = [
    {"role": "user",
     "content": [
         {"type": "text",
          "text": "What's in this image?"
         },
         {"type": "image_url",
          "image_url": {"url": image_url},
         },
    ]},
]
```

これは次のプロンプトになります。

```text
Prompt:
    <s>[INST]What's in this image?\n[IMG][/INST]

Tokenized prompt:
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, 10, 4]

Prompt with placeholders (<P>):
    [1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <P>, <P>, ..., <P>, 4]
```

見てのとおり、トークン化の後、`[IMG]` はプレースホルダートークンの並びに置き換えられ、これらのプレースホルダーはプレフィル中に画像の埋め込みに置き換えられます。プレフィックスキャッシュがこのケースをサポートするうえでの課題は、プレースホルダーから画像を区別する必要があることです。この問題に対処するため、フロントエンドの画像プロセッサが生成した画像のハッシュをエンコードします。たとえば、上のプロンプトのブロックのハッシュは次のようになります（ブロックサイズを 16、プレースホルダートークンを 41 個と仮定）。

```text
Block 0
    Parent hash: None
    Token IDs: 1, 3, 7493, 1681, 1294, 1593, 3937, 9551, <p>, ..., <p>
    Extra hash: <image hash>
Block 1
    Parent hash: Block 0 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 2
    Parent hash: Block 1 hash
    Token IDs: <p>, ..., <p>
    Extra hash: <image hash>
Block 3
    Parent hash: Block 2 hash
    Token IDs: <p>, ..., <p>, 4
    Extra hash: <image hash>
```

このドキュメントの残りでは、まず vLLM v1 のプレフィックスキャッシュで使われるデータ構造を紹介し、続いて主要な KV キャッシュ操作（allocate、append、free、eviction など）におけるプレフィックスキャッシュのワークフローを説明します。最後に、エンドツーエンドのプレフィックスキャッシュのワークフローを例で示します。

**セキュリティのためのキャッシュ分離**
共有環境でのプライバシーを高めるため、vLLM はリクエストごとの任意のソルトによってプレフィックスキャッシュの再利用を分離できます。リクエストに `cache_salt` を含めると、その値が最初のブロックのハッシュに注入され、同じソルトを持つリクエストだけがキャッシュされた KV ブロックを再利用できるようになります。これにより、攻撃者がレイテンシの差を観測してキャッシュされた内容を推測するタイミング攻撃を防げます。性能を損なうことなく保護を提供します。

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Here is a document with details about the world series: ..."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
  "cache_salt": "your-cache-salt"
}
```

この設定により、キャッシュの共有は共通のソルトに明示的に合意したユーザーやリクエストに限定され、信頼グループ内ではキャッシュを再利用しつつ、それ以外からは分離できます。

## データ構造 { #data-structure }

vLLM v1 のプレフィックスキャッシュは KV キャッシュマネージャーに実装されています。基本的な構成要素は「Block」のデータクラスです（簡略版）。

```python
class KVCacheBlock:
    # The block ID (immutable)
    block_id: int
    # The block hash (will be assigned when the block is full,
    # and will be reset when the block is evicted).
    block_hash: BlockHash
    # The number of requests using this block now.
    ref_cnt: int

    # The pointers to form a doubly linked list for the free queue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

特筆すべき設計上のポイントが 2 つあります。

1. KV キャッシュマネージャーの初期化時に、すべての KVCacheBlock をブロックプールとして確保します。これにより Python のオブジェクト生成のオーバーヘッドを避けられ、常にすべてのブロックを簡単に追跡できます。  
2. KVCacheBlock に双方向連結リストのポインタを直接持たせ、free キューをそのまま構成できるようにしています。これには 2 つの利点があります。  
    1. 途中の要素を末尾へ移動する操作を O(1) の計算量で行えます。  
    2. 要素をラップする別の Python のキュー（`deque` など）を導入せずに済みます。

その結果、KV キャッシュマネージャーの初期化時には次のコンポーネントが揃います。

![Component Overview](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/overview.png)

* ブロックプール: KVCacheBlock のリスト。  
* free ブロックキュー: 操作のために先頭と末尾のブロックのポインタのみを保持します。  
* キャッシュブロック: ハッシュキーからブロック ID への対応。  
* リクエストブロック: リクエスト ID から割り当て済みブロック ID への対応。

## 操作 { #operations }

### ブロックの割り当て { #block-allocation }

**新しいリクエスト:** スケジューラが新しいリクエストを KV キャッシュブロックの割り当てとともにスケジュールする流れは次のとおりです。

1. スケジューラは `kv_cache_manager.get_computed_blocks()` を呼び出し、すでに計算済みのブロック列を取得します。これはリクエストのプロンプトトークンをハッシュ化し、キャッシュブロックを検索することで行われます。  
2. スケジューラは `kv_cache_manager.allocate_slots()` を呼び出します。ここでは次の処理を行います。  
    1. 新たに必要なブロック数を計算し、割り当てられるブロックが足りなければリターンします。  
    2. 計算済みブロックを「touch」します。計算済みブロックの参照カウントを 1 増やし、他のリクエストがそのブロックを使っていなければ free キューから取り除きます。これはこれらの計算済みブロックが追い出されるのを防ぐためです。図解は次のセクションの例を参照してください。  
    3. free キューの先頭を取り出して新しいブロックを割り当てます。先頭のブロックがキャッシュされたブロックだった場合、この操作はそのブロックを「追い出し」、以後どのリクエストからも再利用できなくします。  
    4. 割り当てたブロックがすでにトークンで満杯であれば、ただちにキャッシュブロックに追加し、同じバッチ内の他のリクエストから再利用できるようにします。

**実行中のリクエスト:** スケジューラが実行中のリクエストを KV キャッシュブロックの割り当てとともにスケジュールする流れは次のとおりです。

1. スケジューラは `kv_cache_manager.allocate_slots()` を呼び出します。ここでは次の処理を行います。  
    1. 新たに必要なブロック数を計算し、割り当てられるブロックが足りなければリターンします。  
    2. free キューの先頭を取り出して新しいブロックを割り当てます。先頭のブロックがキャッシュされたブロックだった場合、この操作はそのブロックを「追い出し」、以後どのリクエストからも再利用できなくします。  
    3. 既存のブロックと新しいブロックのスロットにトークン ID を追加します。ブロックが満杯になったら、キャッシュブロックに追加してキャッシュします。

**重複したブロック**  
ブロックサイズを 4 とし、プロンプト ABCDEF、デコード長 3 のリクエスト（リクエスト 1）を送るとします。

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 1 (EFG)]
  Cache Blocks: 0
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 1 (EFGH)]
  Cache Blocks: 0, 1
Time 2:
  Tokens: [A, B, C, D, E, F, G, H, I]
  Block Table: [0 (ABCD), 1 (EFGH), 2 (I)]
  Cache Blocks: 0, 1
```

この時点でブロック 0 とブロック 1 がキャッシュされています。ここで同じリクエストを greedy サンプリングでもう一度送ると（リクエスト 2）、リクエスト 1 とまったく同じ出力が得られます。

```text
Prompt: [A, B, C, D, E, F]
Output: [G, H, I]

Time 0:
  Tokens: [A, B, C, D, E, F, G]
  Block Table: [0 (ABCD), 3 (EFG)]
  Cache Blocks: 0, 1
Time 1:
  Tokens: [A, B, C, D, E, F, G, H]
  Block Table: [0 (ABCD), 3 (EFGH)]
  Cache Blocks: 0, 1, 3
```

見てのとおり、ブロック 3 は新しく満杯になったブロックとしてキャッシュされます。しかしこれはブロック 1 と重複しており、同じブロックを 2 回キャッシュしたことになります。v0 では、ブロック 3 の重複を検出するとブロック 3 を解放し、リクエスト 2 に代わりにブロック 1 を使わせていました。そのため Time 1 のブロックテーブルは `[0, 1]` になりました。しかし vLLM v1 のブロックテーブルは追記専用であり、ブロックテーブルを `[0, 3]` から `[0, 1]` に変更することは許されません。その結果、ハッシュキー E-H に対して重複したブロックが存在することになります。この重複は、リクエストが解放されたときに解消されます。

### 解放 { #free }

リクエストが完了すると、他のリクエストが使っていない（参照カウント = 0 の）ブロックをすべて解放します。この例では、リクエスト 1 と、それに紐づくブロック 2、3、4、8 を解放します。解放されたブロックが *逆順* で free キューの末尾に追加されている点に注目してください。これは、リクエストの最後のブロックはより多くのトークンをハッシュに含むため、他のリクエストから再利用される可能性が低いからです。したがって、先に追い出されるべきなのです。

![Free queue after a request is freed](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/free.png)

### 追い出し（LRU） { #eviction-lru }

free キューの先頭ブロック（最も長く使われていないブロック）がキャッシュされている場合、他のリクエストから使われないようにそのブロックを追い出す必要があります。具体的には、追い出しは次の手順で行われます。

1. free キューの先頭からブロックを取り出す。これが追い出し対象の LRU ブロックです。  
2. キャッシュブロックからそのブロック ID を削除する。  
3. ブロックのハッシュを削除する。

## 例 { #example }

この例では、ブロックサイズを 4（各ブロックは 4 トークンをキャッシュできる）とし、KV キャッシュマネージャーには合計 10 個のブロックがあるとします。

**Time 1: キャッシュが空の状態で新しいリクエストが到着。** 4 個のブロックを割り当てます。そのうち 3 個はすでに満杯でキャッシュされます。4 番目のブロックは 4 トークン中 3 トークンで部分的に埋まっています。

![Example Time 1](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/example-time-1.png)

**Time 2: リクエスト 0 がブロック 3 を満杯にし、デコードを続けるために新しいブロックを要求。** ブロック 3 をキャッシュし、ブロック 4 を割り当てます。

![Example Time 2](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/example-time-3.png)

**Time 3: 14 個のプロンプトトークンを持つリクエスト 1 が到着。最初の 10 トークンはリクエスト 0 と同じ。** キャッシュにヒットするのは最初の 2 ブロック（8 トークン）だけです。3 番目のブロックは 4 トークン中 2 トークンしか一致しないためです。

![Example Time 3](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/example-time-4.png)

**Time 4: リクエスト 0 が完了して解放される。** ブロック 2、3、4 が逆順で free キューに追加されます（ただしブロック 2 と 3 はキャッシュされたままです）。ブロック 0 と 1 はリクエスト 1 が使用中のため、free キューには追加されません。

![Example Time 4](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/example-time-5.png)

**Time 5: リクエスト 1 が完了して解放される。**

![Example Time 5](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/example-time-6.png)

**Time 6: 29 個のプロンプトトークンを持つリクエスト 2 が到着。最初の 12 トークンはリクエスト 0 と同じ。** free キュー内のブロックの順序が `7 - 8 - 9 - 4 - 3 - 2 - 6 - 5 - 1 - 0` であっても、キャッシュヒットしたブロック（0、1、2）は割り当ての前に touch されてキューから取り除かれるため、free キューは `7 - 8 - 9 - 4 - 3 - 6 - 5` になります。その結果、割り当てられるブロックは 0（キャッシュ済み）、1（キャッシュ済み）、2（キャッシュ済み）、7、8、9、4、3（追い出し）となります。

![Example Time 6](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/prefix_caching/example-time-7.png)
