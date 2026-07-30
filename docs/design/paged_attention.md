# Paged Attention { #paged-attention }

!!! warning
    これは [vLLM の元論文](https://arxiv.org/abs/2309.06180)にもとづく歴史的な文書です。
    現在の vLLM で使われているコードを説明したものではありません。

現在、vLLM は独自実装のマルチヘッド query attention カーネル（`csrc/attention/attention_kernels.cu`）を使っています。
このカーネルは、key と value のキャッシュを別々のブロックに保存する vLLM のページド KV キャッシュと互換になるよう設計されています（このブロックの概念は GPU のスレッドブロックとは異なる点に注意してください。以降の文書では、vLLM の paged attention のブロックを「ブロック」、GPU のスレッドブロックを「スレッドブロック」と呼びます）。

高い性能を実現するため、このカーネルは特別に設計されたメモリレイアウトとアクセス方法に依存しています。とくに、スレッドがグローバルメモリから共有メモリへデータを読み込む場面で重要です。この文書の目的は、vLLM のマルチヘッド query attention カーネルについて学びたい人の助けとなるよう、カーネルの実装を段階を追って大まかに説明することです。この文書を読み終えた読者は、理解が深まり、実際の実装を追いやすくなるはずです。

なお、この文書は、対応するデータの正しいインデックスの計算方法や内積の実装など、すべての詳細を網羅しているわけではありません。ただし、この文書を読んで大まかな処理の流れに慣れておけば、実際のコードを読んで細部を理解しやすくなるでしょう。

## 入力 { #inputs }

カーネル関数は、現在のスレッドが割り当てられた処理を行うための引数群を受け取ります。最も重要な 3 つの引数は入力ポインタ `q`、`k_cache`、`v_cache` で、これらは読み込んで処理すべき query、key、value のデータをグローバルメモリ上で指します。出力ポインタ `out` は、結果を書き込むべきグローバルメモリを指します。これら 4 つのポインタは実際には多次元配列を指しますが、各スレッドは自分に割り当てられた部分のデータにのみアクセスします。ここでは簡単のため、その他の実行時パラメータはすべて省略しています。

```cpp
template<typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(
    ... // Other side args.
    const scalar_t* __restrict__ out,       // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    ... // Other side args.
)
```

また、関数シグネチャの上には、コンパイル時に決まるテンプレート引数の一覧があります。`scalar_t` は query、key、value のデータ要素のデータ型（FP16 など）を表します。`HEAD_SIZE` は各ヘッドの要素数を示します。`BLOCK_SIZE` は各ブロックのトークン数です。`NUM_THREADS` は各スレッドブロックのスレッド数を表します。`PARTITION_SIZE` はテンソル並列の GPU 数を表します（簡単のため、ここでは 0 でテンソル並列は無効と仮定します）。

これらの引数をもとに、一連の準備処理が必要になります。これには現在のヘッドのインデックスやブロックのインデックス、その他必要な変数の計算が含まれます。ただし今のところ、これらの準備は無視して実際の計算に進んで構いません。全体の流れをつかんだあとのほうが理解しやすくなります。

## 概念 { #concepts }

計算の流れに入る前に、以降の節で必要となるいくつかの概念を説明します。分かりにくい用語に出会ったときに戻ってくることにして、この節は読み飛ばしても構いません。

- **シーケンス**: シーケンスはクライアントのリクエストを表します。たとえば `q` が指すデータは `[num_seqs, num_heads, head_size]` の形状を持ちます。これは `q` が合計 `num_seqs` 個の query シーケンスのデータを指すことを意味します。このカーネルは単一 query の attention カーネルであるため、各シーケンスは query トークンを 1 つだけ持ちます。したがって `num_seqs` は、バッチで処理されるトークンの総数と等しくなります。
- **コンテキスト**: コンテキストは、そのシーケンスで生成されたトークンから成ります。たとえば `["What", "is", "your"]` がコンテキストのトークンで、入力の query トークンが `"name"` です。モデルはトークン `"?"` を生成するかもしれません。
- **vec**: vec は、まとめて読み込まれ計算される要素の並びです。query と key のデータでは、各スレッドグループが一度に 16 バイトのデータを読み込んで計算できるように vec のサイズ（`VEC_SIZE`）が決まります。value のデータでは、各スレッドが一度に 16 バイトのデータを読み込んで計算できるように vec のサイズ（`V_VEC_SIZE`）が決まります。たとえば `scalar_t` が FP16（2 バイト）で `THREAD_GROUP_SIZE` が 2 の場合、`VEC_SIZE` は 4、`V_VEC_SIZE` は 8 になります。
- **スレッドグループ**: スレッドグループは、一度に 1 つの query トークンと 1 つの key トークンを読み込んで計算する少数のスレッド（`THREAD_GROUP_SIZE`）の集まりです。各スレッドはトークンデータの一部だけを担当します。1 つのスレッドグループが処理する要素の総数を `x` と呼びます。たとえばスレッドグループが 2 スレッドを含み、ヘッドサイズが 8 の場合、スレッド 0 はインデックス 0、2、4、6 の query と key の要素を、スレッド 1 はインデックス 1、3、5、7 の要素を担当します。
- **ブロック**: vLLM の key と value のキャッシュデータはブロックに分割されます。各ブロックは、1 つのヘッドについて固定個数（`BLOCK_SIZE`）のトークンのデータを保持します。各ブロックには、コンテキスト全体のトークンのうち一部しか含まれないこともあります。たとえばブロックサイズが 16、ヘッドサイズが 128 の場合、1 つのヘッドについて 1 ブロックは 16 * 128 = 2048 要素を保持できます。
- **ワープ**: ワープは、ストリーミングマルチプロセッサ（SM）上で同時に実行される 32 スレッド（`WARP_SIZE`）の集まりです。このカーネルでは、各ワープが一度に 1 つの query トークンと 1 ブロック全体の key トークンの計算を処理します（複数回の反復で複数ブロックを処理することもあります）。たとえば 1 つのコンテキストに対してワープが 4 つ、ブロックが 6 つある場合、ワープ 0 が 0 番目と 4 番目、ワープ 1 が 1 番目と 5 番目、ワープ 2 が 2 番目、ワープ 3 が 3 番目のブロックを担当する、といった割り当てになります。
- **スレッドブロック**: スレッドブロックは、同じ共有メモリにアクセスできるスレッド（`NUM_THREADS`）の集まりです。各スレッドブロックは複数のワープ（`NUM_WARPS`）を含み、このカーネルでは、各スレッドブロックが 1 つの query トークンとコンテキスト全体の key トークンとの計算を処理します。
- **グリッド**: グリッドはスレッドブロックの集合であり、その集合の形状を定義します。このカーネルでの形状は `(num_heads, num_seqs, max_num_partitions)` です。したがって各スレッドブロックは、1 つのヘッド、1 つのシーケンス、1 つのパーティションについてのみ計算を担当します。

## Query { #query }

この節では、query のデータがどのようにメモリに保存され、各スレッドに読み込まれるかを説明します。前述のとおり、各スレッドグループが 1 つの query トークンのデータを読み込み、各スレッド自身は 1 つの query トークンのデータの一部だけを担当します。ワープ内では、どのスレッドグループも同じ query トークンのデータを読み込みますが、それぞれ異なる key トークンのデータと掛け合わせます。

```cpp
const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
```

![query](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/query.png)

各スレッドは、グローバルメモリ上の割り当てられた query トークンのデータを指す自身の `q_ptr` を定義します。たとえば `VEC_SIZE` が 4、`HEAD_SIZE` が 128 の場合、`q_ptr` は合計 128 要素を含み、128 / 4 = 32 個の vec に分割されたデータを指します。

![q_vecs](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/q_vecs.png)

```cpp
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
```

次に、`q_ptr` が指すグローバルメモリのデータを共有メモリへ `q_vecs` として読み込む必要があります。各 vec が異なる行に割り当てられる点が重要です。たとえば `THREAD_GROUP_SIZE` が 2 の場合、スレッド 0 が 0 行目の vec を、スレッド 1 が 1 行目の vec を担当します。このように query のデータを読み込むことで、スレッド 0 とスレッド 1 のような隣接するスレッドが隣接するメモリを読み込めるようになり、メモリのコアレッシングが実現して性能が向上します。

## Key { #key }

「Query」の節と同様に、この節では key のメモリレイアウトと割り当てを説明します。各スレッドグループは 1 回のカーネル実行で 1 つの query トークンしか扱いませんが、複数回の反復にわたって複数の key トークンを扱うことがあります。同時に、各ワープは複数回の反復で複数ブロックの key トークンを処理し、カーネル実行後にはスレッドグループ全体でコンテキストのすべてのトークンが処理されるようにします。ここでの「扱う」とは、query のデータと key のデータの内積を計算することを指します。

```cpp
const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride
                    + kv_head_idx * kv_head_stride
                    + physical_block_offset * x;
```

`q_ptr` と異なり、各スレッドの `k_ptr` は反復ごとに異なる key トークンを指します。上記のとおり、`k_ptr` は `k_cache` を基点として、割り当てられたブロック・ヘッド・トークンの key トークンデータを指します。

![key](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/key.png)

上図は key のデータのメモリレイアウトを示しています。`BLOCK_SIZE` が 16、`HEAD_SIZE` が 128、`x` が 8、`THREAD_GROUP_SIZE` が 2 で、ワープが合計 4 つある場合を想定しています。各長方形は、1 つのヘッドにおける 1 つの key トークンの全要素を表し、1 つのスレッドグループが処理します。左半分はワープ 0 が担当する合計 16 ブロック分の key トークンデータを、右半分は他のワープや他の反復で扱う残りの key トークンデータを表します。各長方形の内部には合計 32 個の vec（1 トークンあたり 128 要素）があり、2 つのスレッド（1 つのスレッドグループ）が分担して処理します。

![k_vecs](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/k_vecs.png)

```cpp
K_vec k_vecs[NUM_VECS_PER_THREAD]
```

次に、`k_ptr` から key トークンのデータを読み込み、レジスタメモリに `k_vecs` として保存します。`k_vecs` にレジスタメモリを使うのは、これが 1 つのスレッドから 1 度しかアクセスされないのに対し、`q_vecs` は複数のスレッドから何度もアクセスされるためです。各 `k_vecs` は、後の計算のための複数のベクトルを含みます。各 vec は内側の反復ごとに設定されます。この vec の割り当てにより、ワープ内の隣接するスレッドが隣接するメモリをまとめて読めるようになり、ここでもメモリのコアレッシングが促進されます。たとえば、スレッド 0 は vec 0 を、スレッド 1 は vec 1 を読みます。次の内側ループでは、スレッド 0 は vec 2 を、スレッド 1 は vec 3 を読む、といった具合です。

全体の流れがまだ少し分かりにくいかもしれません。心配はいりません。次の「QK」の節を読み進めてください。query と key の計算の流れを、より明快で高い視点から説明します。

## QK { #qk }

以下の疑似コードのとおり、for ループ全体の前に 1 トークン分の query データを読み込み、`q_vecs` に保存します。次に外側の for ループで、異なるトークンを指す複数の `k_ptr` を巡り、内側の for ループで `k_vecs` を用意します。最後に `q_vecs` と各 `k_vecs` の内積を計算します。

```cpp
q_vecs = ...
for ... {
    k_ptr = ...
    for ... {
        k_vecs[i] = ...
    }
    ...
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
}
```

前述のとおり、各スレッドは一度に query と key のトークンデータの一部しか読み込みません。しかし `Qk_dot<>::dot` の内部ではスレッドグループをまたぐ reduction が行われます。したがって、ここで返される `qk` は query と key のトークンの一部同士の内積ではなく、query と key のトークンデータ全体に対する完全な結果です。

たとえば `HEAD_SIZE` が 128、`THREAD_GROUP_SIZE` が 2 の場合、各スレッドの `k_vecs` は合計 64 要素を含みます。しかし返される `qk` は、実際には 128 個の query 要素と 128 個の key 要素の内積の結果です。内積と reduction の詳細を知りたい場合は `Qk_dot<>::dot` の実装を参照してください。ただし簡単のため、この文書では扱いません。

## Softmax { #softmax }

次に、上図のようにすべての `qk` について正規化された softmax を計算します。ここで各 $x$ は 1 つの `qk` を表します。そのためには、すべての `qk` について reduction された `qk_max`（$m(x)$）と `exp_sum`（$\ell(x)$）を求める必要があります。この reduction は、query トークンとコンテキストのすべての key トークンとのあいだの結果を含めて、スレッドブロック全体にわたって行う必要があります。

$$
\begin{gather*}
m(x):=\max _i \quad x_i \\ \quad f(x):=\left[\begin{array}{lll}e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}\end{array}\right]\\ \quad \ell(x):=\sum_i f(x)_i \\
\quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}
\end{gather*}
$$

### `qk_max` と `logits` { #qk_max-and-logits }

`qk` の結果が得られた直後に、暫定の `logits` の値を `qk` で設定できます（最終的に `logits` には正規化された softmax の結果が入ります）。あわせて、現在のスレッドグループが計算したすべての `qk` について `qk_max` を比較して集めることもできます。

```cpp
if (thread_group_offset == 0) {
    const bool mask = token_idx >= context_len;
    logits[token_idx - start_token_idx] = mask ? 0.f : qk;
    qk_max = mask ? qk_max : fmaxf(qk_max, qk);
}
```

ここでの `logits` は共有メモリ上にあるため、各スレッドグループは自分に割り当てられたコンテキストトークンの分の要素を設定します。全体として、logits のサイズはコンテキストトークン数と等しくなります。

```cpp
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}

if (lane == 0) {
    red_smem[warp_idx] = qk_max;
}
```

次に、各ワープ内で reduction した `qk_max` を求める必要があります。基本的な考え方は、ワープ内のスレッド同士が通信して最終的な最大の `qk` を得ることです。

```cpp
for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
}
qk_max = VLLM_SHFL_SYNC(qk_max, 0);
```

最後に、このスレッドブロック内のすべてのワープの `qk_max` を比較することで、スレッドブロック全体で reduction した `qk_max` が得られます。そして、最終結果を各スレッドへブロードキャストする必要があります。

### `exp_sum` { #exp_sum }

`qk_max` と同様に、スレッドブロック全体で reduction した合計値も求める必要があります。

```cpp
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
}
...
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);
```

まず、各スレッドグループのすべての exp 値を合計し、同時に `logits` の各要素を `qk` から `exp(qk - qk_max)` へ変換します。ここでの `qk_max` はすでにスレッドブロック全体での最大の `qk` である点に注意してください。その後、`qk_max` と同様に `exp_sum` についてもスレッドブロック全体で reduction を行えます。

```cpp
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
}
```

最後に、reduction 済みの `qk_max` と `exp_sum` を使って、最終的な正規化 softmax の結果を `logits` として得られます。この `logits` 変数は、後のステップで value のデータとの内積に使われます。この時点で、割り当てられたすべてのコンテキストトークンについて `qk` の正規化 softmax の結果が格納されています。

## Value { #value }

![value](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/value.png)

![logits_vec](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/logits_vec.png)

![v_vec](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/design/paged_attention/v_vec.png)

次に、value のデータを取得して `logits` との内積を計算します。query や key と異なり、value のデータにはスレッドグループの概念がありません。図に示すとおり、key トークンのメモリレイアウトとは異なり、同じ列の要素が同じ value トークンに対応します。1 ブロック分の value データには `HEAD_SIZE` 行と `BLOCK_SIZE` 列があり、複数の `v_vec` に分割されます。

各スレッドは常に、同じ `V_VEC_SIZE` 個のトークンから一度に `V_VEC_SIZE` 個の要素を読み込みます。その結果、1 つのスレッドは複数回の内側の反復を通じて、異なる行・同じ列から複数の `v_vec` を取得します。各 `v_vec` は、対応する `logits_vec`（これも `logits` から取った `V_VEC_SIZE` 個の要素）と内積を取る必要があります。全体として、複数回の内側の反復により各ワープが 1 ブロック分の value トークンを処理し、複数回の外側の反復によりコンテキスト全体の value トークンが処理されます。

```cpp
float accs[NUM_ROWS_PER_THREAD];
for ... { // Iteration over different blocks.
    logits_vec = ...
    for ... { // Iteration over different rows.
        v_vec = ...
        ...
        accs[i] += dot(logits_vec, v_vec);
    }
}
```

上記の疑似コードのとおり、外側のループでは `k_ptr` と同様に `logits_vec` が異なるブロックを巡り、`logits` から `V_VEC_SIZE` 個の要素を読み込みます。内側のループでは、各スレッドが同じトークンから `V_VEC_SIZE` 個の要素を `v_vec` として読み込み、内積を計算します。内側の反復ごとに、スレッドは同じトークンについて異なるヘッド位置の要素を読み込む点が重要です。内積の結果は `accs` に累積されます。したがって `accs` の各要素は、現在のスレッドに割り当てられたヘッド位置に対応します。

たとえば `BLOCK_SIZE` が 16、`V_VEC_SIZE` が 8 の場合、各スレッドは一度に 8 トークン分の value 要素を 8 個読み込みます。各要素は、同じヘッド位置における異なるトークンのものです。`HEAD_SIZE` が 128、`WARP_SIZE` が 32 の場合、内側ループごとにワープは `WARP_SIZE * V_VEC_SIZE = 256` 個の要素を読み込む必要があります。つまり、1 ブロック分の value トークンをワープが処理するには、合計 128 * 16 / 256 = 8 回の内側の反復が必要です。そして各スレッドの `accs` は、8 つの異なるヘッド位置で累積された 8 個の要素を含みます。スレッド 0 の場合、`accs` 変数は 8 個の要素を持ち、それらは value ヘッドの 0 番目、32 番目、…、224 番目の要素が、割り当てられた 8 トークンすべてから累積されたものです。

## LV { #lv }

次に、各ワープ内で `accs` の reduction を行う必要があります。この処理により、各スレッドが 1 ブロック内の全トークンについて、割り当てられたヘッド位置の `accs` を累積できます。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
}
```

次に、すべてのワープにわたって `accs` の reduction を行い、各スレッドがコンテキストの全トークンについて、割り当てられたヘッド位置の `accs` の累積値を持つようにします。各スレッドの `accs` は、コンテキストの全トークンについてヘッド全体のうち一部の要素の累積値しか保持しない点に注意してください。ただし全体としては、出力に必要なすべての結果が計算されており、単に異なるスレッドのレジスタメモリに分散して保持されているだけです。

??? code

    ```cpp
    float* out_smem = reinterpret_cast<float*>(shared_mem);
    for (int i = NUM_WARPS; i > 1; i /= 2) {
        // Upper warps write to shared memory.
        ...
        float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            ...
            dst[row_idx] = accs[i];
        }

        // Lower warps update the output.
        const float* src = &out_smem[warp_idx * HEAD_SIZE];
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
            ...
            accs[i] += src[row_idx];
        }

        // Write out the accs.
    }
    ```

## 出力 { #output }

これで、ローカルのレジスタメモリにある計算結果をすべて、最終的な出力先であるグローバルメモリへ書き出せます。

```cpp
scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                + head_idx * max_num_partitions * HEAD_SIZE
                + partition_idx * HEAD_SIZE;
```

まず、割り当てられたシーケンスとヘッドの開始アドレスを指す `out_ptr` 変数を定義します。

```cpp
for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
    if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
    }
}
```

最後に、割り当てられた各ヘッド位置を巡り、`out_ptr` を基点として対応する累積結果を書き出します。

## 引用 { #citation }

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
