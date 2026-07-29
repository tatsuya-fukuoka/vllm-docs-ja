# 基本的なモデル { #basic-model }

このガイドでは、基本的な vLLM のモデルを実装する手順を説明します。

## 1. モデルのコードを持ち込む { #1-bring-your-model-code }

まず、元のリポジトリから PyTorch のモデルコードをクローンします。たとえば、vLLM の [OPT モデル](../../../vllm/model_executor/models/opt.py)は HuggingFace の [modeling_opt.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py) をもとに作られています。

!!! warning
    元のコードの著作権とライセンス条項を必ず確認し、遵守してください。

## 2. コードを vLLM と互換にする { #2-make-your-code-compatible-with-vllm }

vLLM との互換性を確保するため、モデルは次の要件を満たす必要があります。

### 初期化のコード { #initialization-code }

モデル内のすべての vLLM モジュールは、コンストラクタに `prefix` 引数を持つ必要があります。この `prefix` は通常、モデルの state dict におけるそのモジュールの完全な名前で、次の点で重要です。

- 実行時のサポート: vLLM の Attention 演算子は、完全な名前でモデルの状態に登録されます。衝突を避けるため、各 Attention 演算子は層名として一意の prefix を持つ必要があります。
- 不均一な量子化のサポート: 量子化されたチェックポイントは、一部の層だけを量子化し、他の層を完全な精度のまま保つことができます。初期化時に `prefix` を渡すことで、vLLM は現在の層の `prefix` を量子化の設定と照合し、その層を量子化モードで初期化すべきかを判断できます。

初期化のコードは次のようになります。

??? code

    ```python
    from torch import nn
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention import Attention

    class MyAttention(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.attn = Attention(prefix=f"{prefix}.attn")

    class MyDecoderLayer(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.self_attn = MyAttention(prefix=f"{prefix}.self_attn")

    class MyModel(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str):
            super().__init__()
            self.layers = nn.ModuleList(
                [MyDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}") for i in range(vllm_config.model_config.hf_config.num_hidden_layers)]
            )

    class MyModelForCausalLM(nn.Module):
        def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
            super().__init__()
            self.model = MyModel(vllm_config, prefix=f"{prefix}.model")
    ```

### 計算のコード { #computation-code }

- `MyModel` モジュールに、`input_ids` からテキストの埋め込みを返す `embed_input_ids` メソッドを追加します。これはテキスト埋め込み層を直接呼び出すのと同等ですが、`MyModel` が複合的なマルチモーダルモデルの中で使われる場合に統一的なインターフェースを提供します。

```python
class MyModel(nn.Module):
        ...

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        ... 
```

- モデルの [forward][torch.nn.Module.forward] メソッドを書き換え、学習専用のコードなど不要なものを取り除きます。入力パラメータは、`input_ids` と `positions` を、最大シーケンス長の次元を持たない、単一のバッチサイズ次元を持つ平坦化されたテンソルとして扱うように変更します。

```python
def forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
```

!!! note
    現時点で vLLM は、基本的なマルチヘッド Attention と、rotary positional embedding を用いるその変種を
    サポートしています。モデルが異なる Attention の仕組みを使っている場合は、vLLM に新しい Attention 層を
    実装する必要があります。

参考として、vLLM の [Llama の実装](../../../vllm/model_executor/models/llama.py)を確認してください。vLLM はすでに多数のモデルをサポートしています。自分のモデルに近いモデルを見つけ、それを自分のモデルのアーキテクチャに合わせて修正することを推奨します。その他の例は [vllm/model_executor/models](../../../vllm/model_executor/models) を参照してください。

## 3. （任意）テンソル並列と量子化のサポートを実装する { #3-optional-implement-tensor-parallelism-and-quantization-support }

モデルが大きすぎて 1 台の GPU に収まらない場合は、テンソル並列を使って対応できます。そのためには、モデルの Linear 層と埋め込み層をテンソル並列版に置き換えます。埋め込み層は [torch.nn.Embedding][] を `VocabParallelEmbedding` に置き換えるだけで済みます。出力側の LM ヘッドには `ParallelLMHead` を使えます。Linear 層については、並列化のために次の選択肢を提供しています。

- `ReplicatedLinear`: 入力と重みを複数の GPU に複製します。メモリの削減効果はありません。
- `RowParallelLinear`: 入力テンソルを隠れ次元に沿って分割します。重み行列は行方向（入力次元）に分割されます。行列積のあとに *all-reduce* を実行して結果を集約します。通常、FFN の 2 層目と Attention 層の出力側の線形変換に使われます。
- `ColumnParallelLinear`: 入力テンソルは複製されます。重み行列は列方向（出力次元）に分割されます。結果は列方向に分割されます。通常、FFN の 1 層目と、元の Transformer における Attention 層の分離された QKV 変換に使われます。
- `MergedColumnParallelLinear`: 複数の `ColumnParallelLinear` 演算子を統合した column-parallel の Linear 層です。通常、SiLU などの重み付き活性化関数を伴う FFN の 1 層目に使われます。このクラスは複数の重み行列のシャード化された読み込みロジックを扱います。
- `QKVParallelLinear`: マルチヘッド Attention およびグループ化クエリ Attention の query / key / value の射影のための並列 Linear 層です。key/value ヘッドの数が world size より少ない場合、このクラスは key/value ヘッドを適切に複製します。重みの読み込みと重み行列の複製を扱います。

上記のすべての Linear 層は `linear_method` を入力として受け取る点に注意してください。重みの量子化をサポートするため、vLLM は量子化方式に応じてこのパラメータを設定します。

## 4. 重みの読み込みロジックを実装する { #4-implement-the-weight-loading-logic }

次に、`*ForCausalLM` クラスに `load_weights` メソッドを実装する必要があります。このメソッドは HuggingFace のチェックポイントファイルから重みを読み込み、モデル内の対応する層に割り当てます。特に `MergedColumnParallelLinear` と `QKVParallelLinear` の層については、元のモデルで重み行列が分かれている場合、それぞれの部分を個別に読み込む必要があります。

## 5. モデルを登録する { #5-register-your-model }

新しいモデルを vLLM から使えるように登録する手順は、[このページ](registration.md)を参照してください。

## よくある質問 { #frequently-asked-questions }

### interleaved sliding window を使うモデルへの対応方法 { #how-to-support-models-with-interleaving-sliding-windows }

interleaved sliding window を使うモデルに対応するには、次の点に注意する必要があります。

- モデルの `config.json` に `layer_types` が含まれていることを確認します。
- モデリングのコードで、各層について正しい sliding window の値を解析し、Attention 層の `per_layer_sliding_window` 引数に渡します。参考として[この行](https://github.com/vllm-project/vllm/blob/996357e4808ca5eab97d4c97c7d25b3073f46aab/vllm/model_executor/models/llama.py#L171)を確認してください。

この 2 つの手順で、interleaved sliding window がモデルで動作するようになります。

### Mamba を使うモデルへの対応方法 { #how-to-support-models-that-use-mamba }

3 つのシナリオを考えます。

1. Mamba 層（Mamba-1 または Mamba-2）を使い、Attention 層は使わないモデル。
2. Mamba 層（Mamba-1 または Mamba-2）と Attention 層を組み合わせたモデル。
3. Mamba に似た仕組み（Linear Attention、ShortConv など）と Attention 層を組み合わせたモデル。

ケース (1) では、[`MambaForCausalLM`](../../../vllm/model_executor/models/mamba.py)（Mamba-1 の場合）または [`Mamba2ForCausalLM`](../../../vllm/model_executor/models/mamba2.py)（Mamba-2 の場合）の実装を参考にすることを推奨します。モデルはプロトコル `IsAttentionFree` を継承し、設定から状態の形状とデータ型を計算するクラスメソッド `get_mamba_state_dtype_from_config` と `get_mamba_state_shape_from_config` も実装する必要があります。Mamba 層そのものには、[`MambaMixer`](../../../vllm/model_executor/layers/mamba/mamba_mixer.py)（Mamba-1）または [`MambaMixer2`](../../../vllm/model_executor/layers/mamba/mamba_mixer2.py)（Mamba-2）のクラスを使ってください。実行時の既定値が最適化されるよう、モデルを [vllm/model_executor/models/config.py](../../../vllm/model_executor/models/config.py) の `MODELS_CONFIG_MAP` 辞書にも追加してください。

ケース (2) では、[`JambaForCausalLM`](../../../vllm/model_executor/models/jamba.py)（Mamba-1 と Attention を併用するモデルの例）や [`NemotronHForCausalLM`](../../../vllm/model_executor/models/nemotron_h.py)（Mamba-2 と Attention を併用するモデルの例）の実装を参考にすることを推奨します。これらのモデルはケース (1) と同じ手順に従いますが、（`IsAttentionFree` ではなく）プロトコル `IsHybrid` を継承する必要があり、`MODELS_CONFIG_MAP` への追加は*不要*です（実行時の既定値はプロトコルから推測されます）。

ケース (3) では、カスタムの「Mamba に似た」層 `ShortConv` を使う [`Lfm2ForCausalLM`](../../../vllm/model_executor/models/lfm2.py) の実装を参考にすることを推奨します。これらのモデルの実装にはケース (2) と同じ指針に従ってください。ここでいう「Mamba に似た」とは、（Attention の KV キャッシュのように）追記されるのではなく、その場で更新される状態を持つ層を指します。新しいカスタムの Mamba に似た層を実装する場合は、`MambaBase` を継承し、実行時にデータ型と状態の形状を計算する `get_state_dtype`、`get_state_shape` の各メソッドに加え、`mamba_type` と `get_attn_backend` も実装してください。すべての層に共通するメタデータを扱う「attention メタデータ」クラスの実装も必要です。その例として [`LinearAttentionMetadata`](../../../vllm/v1/attention/backends/linear_attn.py) や [`ShortConvAttentionMetadata`](../../../vllm/v1/attention/backends/short_conv_attn.py) を参照してください。新しい Mamba のバックエンドを追加する際は、[`registry.py`](../../../vllm/v1/attention/backends/registry.py) の `MambaAttentionBackendEnum` も更新すべき点に注意してください。最後に、torch compile と CUDA graph をサポートしたい場合は、Mamba に似た層への呼び出しをカスタム op で包み、それを登録する必要があります。その例は [vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py](../../../vllm/model_executor/layers/mamba/linear/minimax_linear_attn.py) や [vllm/model_executor/layers/mamba/short_conv.py](../../../vllm/model_executor/layers/mamba/short_conv.py) の `direct_register_custom_op` の呼び出しを参照してください。その後、piecewise CUDA graph が意図どおり動作するよう、新しいカスタム op を [vllm/config/compilation.py](../../../vllm/config/compilation.py) の `_attention_ops` のリストに追加してください。
