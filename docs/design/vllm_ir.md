# vLLM IR: 関数型の中間表現 { #vllm-ir-functional-intermediate-representation }

## 動機 { #motivation }

vLLM IR は、低レベルの `torch` op と、`RMSNorm` や量子化演算子のような vLLM の層とのあいだのギャップを埋める**関数型の中間表現（IR）**です。演算子の**意味論**を**実装**および**ディスパッチ**から分離することで、vLLM IR はコンパイルとカーネルの登録・ディスパッチの両方を同時に簡素化します。torch FX 表現における**方言（dialect）**として動作するため、「通常の」torch op やカスタムの torch op / カーネルと完全に相互運用でき、従来の `CustomOp` 方式からの段階的な移行も可能です。

主要な設計原則:

- **eager とコンパイルの一貫性**: eager モードとコンパイルモードで（わずかな数値差を除き）同一の挙動
- **単純で透明、それでいて強力なカーネル選択**: 見通しと制御性が高く、デバッグが容易
- **設定より規約**: op と実装の登録に必要な定型コードがほぼゼロ
- **拡張性**: op も実装も、in-tree / out-of-tree を問わずどこからでも登録可能
- **相互運用性**: 「通常の」torch op やカスタムの torch op / カーネルと完全に互換であり、開発者の負担を減らし段階的な移行を可能にする

意味論と実装をきれいに分離することで、統一的で拡張可能なディスパッチ機構が実現し、プラットフォームごとに複数のカーネルを持てるようになり、強力なカーネル選択が可能になります。この分離により、テストとベンチマークも整理され、従来手法にありがちだった定型コードの多くが不要になります。

カーネル選択をコンパイル処理の後半まで遅らせることで、コンパイラはより高水準の表現の上で動作でき、次の主な利点が得られます。

- 融合 / 変換パスにおけるパターンマッチが、op ごとに単一の単純なパターンで済む
- out-of-tree のコンパイラバックエンドが、より高水準の表現から lowering できる（作業中）
- コンパイラが利用可能な実装の中から自動チューニングできる（将来の機能）

## 概要 { #quick-overview }

### IR 演算の宣言 { #declaring-an-ir-operation }

IR 演算は、その op の意味論を定義するネイティブな PyTorch 実装とともに `@register_op` デコレータで宣言します。

```python
# vllm/ir/ops/layernorm.py
from torch import Tensor
from vllm.ir import register_op

@register_op
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x
```

ネイティブ実装は次の 3 つの役割を果たします。

1. **意味論の定義**: 形状やストライドを含め、その演算の正確な意味論を規定します
2. **既定の実装**: 他の（より優れた）実装が利用できない場合に使われます
3. **テストの基準**: 他の実装はこの意味論に一致している必要があります

### 実装の登録 { #registering-implementations }

カーネルの実装は、IR op オブジェクトの `register_impl` デコレータで登録します。

```python
# vllm/kernels/vllm_c.py
from vllm import ir

rms_norm_no_var = lambda x, weight, epsilon, variance_size=None: variance_size is None

@ir.ops.rms_norm.register_impl("vllm_c", supports_args=rms_norm_no_var, supported=current_platform.is_cuda_alike())
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None) -> Tensor:
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output
```

実装では次を指定できます。

- `supported`: この実装が利用可能かどうかを示す静的な真偽値
- `supports_args`: 特定の引数をこの実装がサポートするかを判定する関数
- `inplace`: この実装が出力のために入力のメモリを再利用するかどうか

### モデル内での IR 演算の利用 { #using-ir-operations-in-models }

IR 演算はモデルのコードで直接 import して呼び出します。

```python
# vllm/model_executor/layers/layernorm.py
from vllm import ir

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: Tensor, residual: Tensor | None = None):
        if residual is None:
            return ir.ops.rms_norm(x, self.weight, self.variance_epsilon)

        # Use maybe_inplace overload to allow implementation to reuse input memory for outputs
        # (using x or residual after this call is undefined behavior)
        return ir.ops.fused_add_rms_norm.maybe_inplace(
            x, residual, self.weight, self.variance_epsilon
        )
```

### カーネル選択の設定 { #configuring-kernel-selection }

カーネルの選択は、設定内の優先順位リストで制御します。優先順位リストは実装を検討する順序を指定し、最初にサポートされている実装が選ばれます。ここでの判定には、静的なサポート確認（`supported=...`）と動的な引数サポート確認（`supports_args=...`）の両方が含まれます。

#### コマンドラインでの設定 { #command-line-configuration }

`--ir-op-priority.<op_name>=<provider1>,<provider2>,...` を使います。

```bash
# CUDA: Use vllm_c implementation for rms_norm
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c

# ROCm: Try aiter first, fall back to vllm_c, then native
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=aiter,vllm_c,native

# Configure multiple operations
vllm serve meta-llama/Llama-3.2-1B \
  --ir-op-priority.rms_norm=vllm_c \
  --ir-op-priority.fused_add_rms_norm=vllm_c
```

#### Python での設定 { #python-configuration }

```python
from vllm import LLM
from vllm.config import VllmConfig, KernelConfig

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    vllm_config=VllmConfig(
        kernel_config=KernelConfig(
            ir_op_priority={
                "rms_norm": ["vllm_c", "native"],
                "fused_add_rms_norm": ["vllm_c", "native"],
            }
        )
    )
)
```

#### プラットフォームの既定値 { #platform-defaults }

各プラットフォームは、自動的に適用される既定の優先順位リストを提供します。

```python
# CUDA/XPU/ROCm platform defaults (when compiling with Inductor)
{
  "rms_norm": ["native"],  # Native torch is default
  "fused_add_rms_norm": ["native"],
}

# CUDA platform defaults (eager or Dynamo-only)
{
  "rms_norm": ["vllm_c", "native"],
  "fused_add_rms_norm": ["vllm_c", "native"],
}

# ROCm platform defaults (future - currently same as CUDA)
{
    "rms_norm": ["aiter", "vllm_c", "native"],
    "fused_add_rms_norm": ["aiter", "vllm_c", "native"],
}

# XPU platform defaults (eager or Dynamo-only)
{
    "rms_norm": ["xpu_kernels", "native"],
    "fused_add_rms_norm": ["xpu_kernels", "native"],
}
```

ユーザーが指定した優先順位はプラットフォームの既定値の前に追加されるため、順序を変えたい実装だけを指定すれば済みます。それ以外の実装は自動的に後ろに追加されます。

## コンパイルのパイプライン { #compilation-pipeline }

vLLM IR は、`torch.compile` にもとづくコンパイル処理を大きくカスタマイズし、カスタムのコンパイルパスが高水準の IR 上で動作しつつ、最終的には効率的な低水準コードを生成できるようにしています。コンパイルのパイプラインは複数の段階から構成されます。

### 1. Dynamo によるトレース { #1-dynamo-tracing }

`torch.compile` がモデルの forward pass をトレースするとき、vLLM IR の演算は `vllm_ir` という torch ライブラリのカスタム演算として現れます。これらの演算は Dynamo からは不透明であり、分解されずに FX グラフへ直接現れます。

```python
# Python code (epsilon=1e-5)
x1 = ir.ops.rms_norm(x, weight, epsilon)
x2, residual_out = ir.ops.fused_add_rms_norm.maybe_inplace(x1, residual, weight, epsilon)

# FX graph after Dynamo tracing
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

### 2. AOTAutograd と関数化 { #2-aotautograd-and-functionalization }

AOTAutograd はグラフを関数化し、状態を変更する演算を関数的に等価なものへ変換します。`maybe_inplace` オーバーロードを持つ vLLM IR の演算については、pre-grad のカスタムパスのフックを使い、AOTAutograd の前に手動でこの変換を行い、関数的な `default` オーバーロードへ置き換えます。

```python
# After functionalization
x1 = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5); x = None
out = torch.ops.vllm_ir.fused_add_rms_norm.default(x1, residual, weight, 1e-5); x1 = residual = None
x2 = out[0]
residual_out = out[1]
```

このパスは、どの入力が「譲渡された（donated）」か（`maybe_inplace` に渡されたか）も追跡し、その情報を vLLM の `PassContext` に保存して、後の clone 除去で利用します。

### 3. IR の融合と変換のパス { #3-ir-fusion-and-transformation-passes }

関数化のあと、vLLM のカスタムパスが高水準の IR 演算を含む関数的な FX グラフの上で動作します。これらのパスは融合、シーケンス並列のための演算の分散、その他の変換を行えます。

```python
# Example: Sequence Parallelism (see SequenceParallelismPass)
# Before SP pass

all_reduce = torch.ops.vllm.all_reduce(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)

# after SP pass
reduce_scatter = torch.ops.vllm.reduce_scatter(x, "tp:0")
rms_norm = torch.ops.vllm_ir.rms_norm(all_reduce, weight, 1e-5)
all_gather = torch.ops.vllm.all_gather(x, "tp:0")
```

融合のパスは高水準の表現から恩恵を受けます。低水準の PyTorch 演算に対してマッチさせる必要も、異なるカーネル実装を個別に扱う必要も、カスタムカーネルの関数化に対処する必要もありません。

### 4. IR の lowering { #4-ir-lowering }

lowering のパス（`VllmIRLoweringPass`）は、各 vLLM IR 演算を選択された実装へ置き換えます。実装は、op の引数の代わりにグラフのメタデータにある **fake テンソル**を使い、優先順位リストとサポート判定にもとづいて選ばれます。

```python
# Implementation selection, same in eager dispatch and compile lowering
def dispatch(*args) -> IrOpImpl:
  for provider in priority_list:  # e.g., ["vllm_c", "native"]
    impl = ir_op.impls[provider]
    if not impl.supported:
      continue
    if impl.supports_args and not impl.supports_args(*args):
      continue
    return impl

# make_fx uses torch.fx.symbolic_trace
impl_graph = make_fx(selected_impl.impl_fn)
# Replace IR op node with impl_graph's nodes
match.replace_by_example(selected_impl.impl_fn, node.args)
```

たとえば、`vllm_c` の実装で `rms_norm` を lowering すると次のようになります。

```python
# Before lowering (IR op)
rms_norm = torch.ops.vllm_ir.rms_norm.default(x, weight, 1e-5)

# After lowering (vllm_c implementation traced)
# Note: Lowering does not currently functionalize, this will likely change in the future.
empty =  torch.ops.aten.empty.memory_format(x.shape, ...)
rms_norm = torch.ops._C.rms_norm(empty, x, weight, 1e-5)
```

入力を変更する実装（`inplace=True`）を lowering する場合、lowering のパスは関数的な意味論を保つために clone を挿入します。

```python
# vllm_c implementation for fused_add_rms_norm mutates its first two arguments
# Lowered with clones for safety
clone_default = torch.ops.aten.clone.default(x)
clone_default_1 = torch.ops.aten.clone.default(residual)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(clone_default, clone_default_1, weight, 1e-5)
```

### 5. clone のクリーンアップ { #5-clone-cleanup }

lowering のあと、clone 除去のパス（`UnsafeCloneEliminationPass`）が lowering 中に挿入された不要な clone を取り除きます。このパスは、`maybe_inplace` とともに in-place カーネルを使う際にゼロコピーの挙動を実現するうえで不可欠です。次の場合に clone が除去されます。

- clone された入力がグラフ内で生成され、その後グラフ内で再利用されていない
- clone された入力がグラフのパラメータであり、譲渡されたものとして印が付いている

```python
# After cleanup (donated inputs, no subsequent uses)
fused_add_rms_norm = torch.ops._C.fused_add_rms_norm.default(x, residual, weight, 1e-5)
```

inplace の関数化（譲渡された入力の追跡）と clone のクリーンアップを組み合わせることで、コンパイラは冗長なコピーを増やしたりメモリ使用量を増やしたりすることなく、in-place カーネルを安全に利用できます。

### 6. Inductor による最適化とコード生成 { #6-inductor-optimization-and-codegen }

IR の lowering とクリーンアップのあと、グラフには標準的な PyTorch の演算とプラットフォーム固有のカスタム op のみが残ります。その後 Inductor が標準のコード生成を行います。

- **Inductor の lowering と pointwise 融合**: 要素ごとの演算や reduction などを融合します
- **メモリプランニング**: バッファの確保と再利用を決定します
- **カーネル生成**: 融合された演算に対して Triton や C++ のコードを生成します
- **オートチューニング**: 最適なカーネル設定を選択します

### パイプラインのまとめ { #pipeline-summary }

```text
モデルの forward pass
    ↓
[Dynamo によるトレース] → vllm_ir.* の op を含む FX グラフ
    ↓
[pre-grad: inplace の関数化] → maybe_inplace → default、譲渡された入力を追跡
    ↓
[AOTAutograd] → 関数化
    ↓
[post-grad: IR の融合パス] → 高水準の IR op を融合（rms_norm + quant など）
    ↓
[post-grad: IR の lowering] → vllm_ir.* の op → 実装の op（必要なら clone 付き）
    ↓
[post-grad: clone のクリーンアップ] → 譲渡された入力の情報を使って不要な clone を除去
    ↓
[Inductor] → パターンマッチ、融合、メモリプランニング、コード生成
    ↓
コンパイル済みコード
```

## vLLM IR の中核的な概念 { #core-vllm-ir-concepts }

### 演算の宣言 { #operation-declaration }

演算は `@register_op` デコレータで宣言し、`IrOp` オブジェクトが生成されます。

```python
@register_op(
    name=None,           # Operation name (defaults to function name)
    activations=None,    # List of activation parameters (defaults to params starting with 'x')
    allow_inplace=False, # Whether to create a maybe_inplace overload
)
def op_name(...):
    ...
```

**パラメータ:**

- `activations`: 「アクティベーション」とみなすパラメータ名のリスト（通常は `maybe_inplace` によって消費されます）。既定では `x` で始まるパラメータです。
- `allow_inplace`: メモリ効率の良い実行のための `maybe_inplace` オーバーロードを作成します（下記参照）。

### `maybe_inplace` オーバーロード { #the-maybe_inplace-overload }

`maybe_inplace` オーバーロードは、LLM 推論のメモリ効率にとってきわめて重要な機能です。これは、演算のあとに呼び出し側がアクティベーションの入力を保持する必要がないことを示し、in-place な実装が出力のために入力のメモリを再利用できるようにします。

#### 意味論と使い方 { #semantics-and-usage }

```python
# Standard usage: inputs are preserved
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
# x and residual are unchanged, out and res_out are new tensors

# maybe_inplace: inputs may be modified
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
# x and residual may be modified (undefined behavior to use them after this)
# out and res_out may alias x and residual
```

`maybe_inplace` に渡したあとでアクティベーションの入力を使うことは**未定義動作**です。

```python
# WRONG: Using x after donating it
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight, epsilon)
result = out + x  # ERROR: x was donated!
```

入力を保持する必要がある場合は、既定のオーバーロードを使うか、手動で clone してください。

```python
# Option 1: Use default overload
out, res_out = ir.ops.fused_add_rms_norm(x, residual, weight, epsilon)
result = out + x  # OK: x is preserved

# Option 2: Clone before maybe_inplace
out, res_out = ir.ops.fused_add_rms_norm.maybe_inplace(x.clone(), residual, weight, epsilon)
result = out + x  # OK: x is preserved, clone was donated
```

#### コンパイル時の挙動 { #compilation-behavior }

コンパイル時、inplace 関数化のパスは譲渡された入力が再利用されていないことを検証し、`maybe_inplace` を関数的な `default` オーバーロードへ変換します。

```python
# Inplace functionalization pass (pre-grad)
for node in graph.nodes:
    if node.target == torch.ops.vllm_ir.fused_add_rms_norm.maybe_inplace:
        # Check that activation inputs aren't used after this node
        for activation_arg in activation_inputs:
            for user in activation_arg.users:
                if user appears after node:
                    raise ValueError(f"Input {activation_arg} donated but used again")

        # Convert to default overload
        node.target = torch.ops.vllm_ir.fused_add_rms_norm.default

        # Track donated graph inputs for later clone elimination
        for i, arg in enumerate(node.args):
            if arg.op == "placeholder" and i in activation_indices:
                pass_context.donated_input_ids.add(node_to_idx[arg])
```

譲渡された入力の情報は、その後 clone クリーンアップのパスで使われ、in-place カーネルが lowering される際の不要なコピーを除去します。

#### eager モードでの挙動 { #eager-mode-behavior }

eager モード（`torch.compile` なし）では、`maybe_inplace` により IR 演算が in-place な実装へ直接ディスパッチできるようになり、**最大限にメモリ効率の良い**実行が可能になります。

```python
# Eager dispatch logic for maybe_inplace
impl: IrOpImpl = ir_op.dispatch(*args)
return impl.impl_fn(*args)

# Eager dispatch logic for default:
impl: IrOpImpl = ir_op.dispatch(*args)
if impl.inplace:
  args = [
    arg.clone() if i in ir_op.activations else arg
    for i, arg in enumerate(args)
  ]
return impl.impl_fn(*args)
```

モデルコードでの `maybe_inplace` と in-place なカーネル実装を組み合わせることで、eager モードとコンパイルモードのどちらでも最適なメモリ効率が得られ、しかも両者の意味論は同一です。

#### メモリ削減の例 { #memory-savings-example }

残差接続を持つ transformer 層を考えます。

```python
# Without maybe_inplace (2 allocations per layer)
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm(hidden_states, input, weight, eps)
# Memory: input (preserved), hidden_states (preserved), normed (new), residual (new)

# With maybe_inplace (0 allocations per layer when using in-place kernel)
hidden_states = self.attention(input)
normed, residual = ir.ops.fused_add_rms_norm.maybe_inplace(hidden_states, input, weight, eps)
# Memory: normed (reuses hidden_states), residual (reuses input)
```

### 実装の登録 { #implementation-registration }

実装は `register_impl` メソッドで登録します。

```python
@ir.ops.op_name.register_impl(
    provider="provider_name",  # Unique identifier (e.g., "vllm_c", "aiter", "triton")
    supported=True,            # Static availability check
    supports_args=None,        # Dynamic argument support check
)
def impl_fn(...):
    ...
```

**プロバイダ名の命名規約:**

- `native`: ネイティブな torch 実装（`@register_op` で宣言）のために予約されています
- `vllm_c`: `torch.ops._C` 経由の C++ / CUDA カーネル
- `aiter`: AMD の AITER ライブラリ
- `xpu_kernels`: `vllm-xpu-kernels` で実装された SYCL / SYCLTLA カーネル
- `triton_*`: Triton カーネル
- その他の実装ではプラットフォーム名 / ライブラリ名

**サポートの確認:**

- `supported`: 静的な真偽値で、import 時に一度だけ評価されます（`HAS_TRITON`、`is_cuda_alike()` など）
- `supports_args`: 引数の互換性を判定する関数 `(*args, **kwargs) -> bool`
    - コンパイル時にはコストゼロで判定するため **fake テンソル**とともに呼ばれます
    - eager モードのディスパッチ時には**実テンソル**とともに呼ばれます
    - バッチサイズを確認したり、値にもとづく guard を追加したりしては**いけません**

サポート判定の例:

```python
def aiter_rms_norm_supports(x, weight, epsilon, variance_size=None):
    # Check dtype (OK: doesn't depend on batch size)
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return False
    # Check optional parameter (OK: static check)
    if variance_size is not None:
        return False
    return True

@ir.ops.rms_norm.register_impl("aiter", supports_args=aiter_rms_norm_supports)
def rms_norm(...):
    ...
```

`VLLM_BATCH_INVARIANT=1` が設定されている場合、バッチ不変（batch-invariant）なカーネルが自動的に選択されます。

### eager モードとコンパイルモード { #eager-mode-vs-compile-mode }

vLLM IR の演算は、eager モードとコンパイルモードで同一の挙動を示します。

**eager モード:**

- 優先順位リストにもとづいて実装へ直接ディスパッチ
- サポート判定は実テンソルの引数で実施
- オーバーヘッドは最小限（必要ならさらに最適化可能）

**コンパイルモード:**

- IR の op は `torch.ops.vllm_ir.*` のカスタム op として FX グラフに現れる
- lowering が fake テンソルを使って実装を選択
- Inductor の最適化と完全に統合

この一貫性により、次のことが可能になります。

- eager モードで安心してプロトタイピングできる
- コンパイルを無効にしてデバッグできる
- eager 実行からコンパイル実行へ段階的に移行できる

## その他のトピック { #other-topics }

### out-of-tree の実装 { #out-of-tree-implementations }

外部のプラットフォームは、vLLM を変更することなく実装を登録できます。

```python
# In external package
from vllm import ir

@ir.ops.rms_norm.register_impl("my_platform", supported=is_my_platform())
def rms_norm(x, weight, epsilon, variance_size=None):
    return my_platform.rms_norm(x, weight, epsilon)
```

そのうえで、自分の実装を使うよう優先順位を設定します。

```python
class MyPlatform(Platform):
  def get_default_ir_op_priority(self):
    return IrOpPriorityConfig(rms_norm=['my_platform', 'native'])

# Users can still override priority in the same way
llm = LLM(ir_op_priority=IrOpPriorityConfig(rms_norm=['custom_oot_kernel']))
```

### デバッグと可観測性 { #debugging-and-observability }

!!! note
    あなたのユースケースで可観測性をどう改善できるか、ぜひ教えてください。

カーネルの選択を確認するには、デバッグログを有効にします。

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ...
```

これにより次がログ出力されます。

- 各演算についてどの実装が選択されたか
- 実装が却下された理由（未サポート、引数が非対応）
- コンパイルキャッシュのヒット / ミス
- IR の lowering の統計

コンパイル済みグラフで選択された実装を確認するには次のようにします。

```python
# After compilation, inspect the lowering pass
lowering_pass = backend.lowering_pass
print(lowering_pass.selected_impls)
# Output: {'rms_norm': {'node_123': 'vllm_c', 'node_456': 'vllm_c'}}
```

## CustomOp からの移行 { #migration-from-customop }

vLLM IR は `CustomOp` と共存し、段階的に置き換えるよう設計されています。

1. **op の宣言**: `CustomOp` クラスを `PluggableLayer` に変換し、`forward_native` を `@register_op` の関数へ移します
2. **実装の登録**: メソッドをオーバーライドする代わりに `@ir.ops.op_name.register_impl` を使います
3. **層での利用**: `self.op(...)` を `ir.ops.op_name(...)` に置き換えます
4. **設定**: `--compilation-config.custom-ops` を `--ir-op-priority` へ移行します

移行は 1 演算ずつ、段階的に進められます。

## 関連項目 { #see-also }

- [torch.compile の統合](torch_compile.md) - 一般的なコンパイル基盤
- [Fusions](fusions.md) - vLLM のカスタム融合・変換パス
- [カスタム演算](custom_op.md) - 従来のカスタム op システム
