# サポートされるモデル { #supported-models }

vLLM は、さまざまなタスクにわたって[生成](./generative_models.md)モデルと[プーリング](./pooling_models/README.md)モデルをサポートしています。

タスクごとに、vLLM で実装済みのモデルアーキテクチャを一覧にしています。
各アーキテクチャには、それを利用している代表的なモデルもあわせて記載しています。

## モデルの実装 { #model-implementation }

### vLLM

vLLM がモデルをネイティブにサポートしている場合、その実装は [vllm/model_executor/models](../../vllm/model_executor/models) にあります。

これらのモデルが、[サポートされるテキストモデル](#list-of-text-only-language-models)および[サポートされるマルチモーダルモデル](#list-of-multimodal-language-models)に一覧されているものです。

### Transformers

vLLM は Transformers で利用できるモデル実装もサポートしています。この機能を「Transformers モデリングバックエンド」と呼びます。Transformers モデリングバックエンドで読み込んだモデルの性能は、vLLM 専用のモデル実装と同等になるはずです。

現時点で、Transformers モデリングバックエンドは以下に対応しています。

- モダリティ: 埋め込みモデル、言語モデル、視覚言語モデル*
- アーキテクチャ: エンコーダのみ、デコーダのみ、mixture-of-experts
- attention の種類: full attention および / または sliding attention

_*視覚言語モデルは現時点で画像入力のみを受け付けます。動画入力のサポートは今後のリリースで追加されます。_

Transformers のモデル実装が[カスタムモデルの記述](#writing-custom-models)の手順をすべて満たしていれば、Transformers モデリングバックエンドで使ったときに vLLM の次の機能と互換になります。

- [互換性マトリクス](../features/README.md#feature-x-feature)に記載されたすべての機能
- 以下の vLLM の並列化方式の任意の組み合わせ
    - データ並列
    - テンソル並列
    - エキスパート並列
    - パイプライン並列

モデリングバックエンドが Transformers かどうかは、次のように簡単に確認できます。

```python
from vllm import LLM
llm = LLM(model=...)  # Name or path of your model
llm.apply_model(lambda model: print(type(model)))
```

表示された型が `Transformers...` で始まっていれば、Transformers のモデル実装が使われています。

vLLM の実装があるモデルでも、Transformers モデリングバックエンド経由で Transformers の実装を使いたい場合は、[オフライン推論](../serving/offline_inference.md)では `model_impl="transformers"` を、[オンラインサービング](../serving/online_serving/README.md)では `--model-impl transformers` を指定してください。

!!! note
    視覚言語モデルでは、`dtype="auto"` で読み込む場合、config に `dtype` があれば vLLM はモデル全体をその `dtype` で読み込みます。一方、ネイティブの Transformers はモデル内の各バックボーンの `dtype` 属性を尊重します。そのため性能にわずかな差が生じることがあります。

#### カスタムモデル { #custom-models }

vLLM でも Transformers でもネイティブにサポートされていないモデルでも、vLLM で使うことができます。

モデルを vLLM の Transformers モデリングバックエンドと互換にするには、次を満たす必要があります。

- Transformers 互換のカスタムモデルであること（[Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models) を参照）
    - モデルディレクトリが正しい構造になっていること（例: `config.json` が存在する）。
    - `config.json` に `auto_map.AutoModel` が含まれること。
- vLLM の Transformers モデリングバックエンドと互換なモデルであること（[カスタムモデルの記述](#writing-custom-models)を参照）
    - カスタマイズはベースモデル側で行うこと（例: `MyModelForCausalLM` ではなく `MyModel`）。

互換なモデルが次の場所にある場合は、

- Hugging Face Model Hub 上にある場合は、[オフライン推論](../serving/offline_inference.md)では `trust_remote_code=True` を、[オンラインサービング](../serving/online_serving/README.md)では `--trust-remote-code` を指定するだけです。
- ローカルディレクトリにある場合は、[オフライン推論](../serving/offline_inference.md)では `model=<MODEL_DIR>` にディレクトリパスを渡し、[オンラインサービング](../serving/online_serving/README.md)では `vllm serve <MODEL_DIR>` とするだけです。

つまり vLLM の Transformers モデリングバックエンドを使えば、Transformers や vLLM で正式にサポートされる前の新しいモデルを利用できます。

#### カスタムモデルの記述 { #writing-custom-models }

この節では、Transformers 互換のカスタムモデルを vLLM の Transformers モデリングバックエンドと互換にするために必要な変更点を説明します（Transformers 互換のカスタムモデルはすでに作成済みであることを前提とします。[Transformers - Customizing models](https://huggingface.co/docs/transformers/en/custom_models) を参照してください）。

モデルを Transformers モデリングバックエンドと互換にするには、次が必要です。

1. `MyModel` から `MyAttention` まで、すべてのモジュールを通して `kwargs` を渡すこと。
    - モデルがエンコーダのみの場合
        1. `MyAttention` に `is_causal = False` を追加します。
    - モデルが mixture-of-experts（MoE）の場合
        1. sparse MoE ブロックに `experts` という属性が必要です。
        2. `experts` のクラス（`MyExperts`）は次のいずれかである必要があります。
            - `nn.ModuleList` を継承する（naive）。
            - あるいは 3D の `nn.Parameters` のみを持つ（packed）。
        3. `MyExperts.forward` は `hidden_states`、`top_k_index`、`top_k_weights` を受け取る必要があります。
2. `MyAttention` は attention の呼び出しに `ALL_ATTENTION_FUNCTIONS` を使う必要があります。
3. `MyModel` に `_supports_attention_backend = True` が含まれている必要があります。

<details class="code">
<summary>modeling_my_model.py</summary>

```python

from transformers import PreTrainedModel
from torch import nn

class MyAttention(nn.Module):
    is_causal = False  # Only do this for encoder-only models

    def forward(self, hidden_states, **kwargs):
        ...
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            **kwargs,
        )
        ...

# Only do this for mixture-of-experts models
class MyExperts(nn.ModuleList):
    def forward(self, hidden_states, top_k_index, top_k_weights):
        ...

# Only do this for mixture-of-experts models
class MySparseMoEBlock(nn.Module):
    def __init__(self, config):
        ...
        self.experts = MyExperts(config)
        ...

    def forward(self, hidden_states: torch.Tensor):
        ...
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        ...

class MyModel(PreTrainedModel):
    _supports_attention_backend = True
```

</details>

このモデルを読み込むと、背後では次のことが起こります。

1. config が読み込まれます。
2. config の `auto_map` から `MyModel` の Python クラスが読み込まれ、そのモデルが `is_backend_compatible()` であることを確認します。
3. `MyModel` は [vllm/model_executor/models/transformers](../../vllm/model_executor/models/transformers) にある Transformers モデリングバックエンドのクラスのいずれかに読み込まれ、そこで `self.config._attn_implementation = "vllm"` が設定されて vLLM の attention 層が使われるようになります。

以上です。

モデルを vLLM のテンソル並列やパイプライン並列の機能と互換にするには、モデルの config クラスに `base_model_tp_plan` や `base_model_pp_plan` を追加する必要があるかもしれません。

<details class="code">
<summary>configuration_my_model.py</summary>

```python

from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):
    base_model_tp_plan = {
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

</details>

- `base_model_tp_plan` は、完全修飾の層名のパターンをテンソル並列のスタイル（現時点では `"colwise"` と `"rowwise"` のみサポート）へ対応づける `dict` です。
    - 標準的な attention（`q`/`k`/`v`/`o_proj`）とゲート付き MLP / experts（`gate`/`up`/`down_proj`）の射影については、融合できる場合に vLLM がテンソル並列のスタイルを推定するため、記載が不要なこともあります。`base_model_tp_plan` が_必要_なのは、これらのパターンに従わない層だけです。融合もされずプランにも記載されていない線形層は複製されます。
- `base_model_pp_plan` は、直下の子層の名前を、`str` の `list` の `tuple` へ対応づける `dict` です。
    - これが必要なのは、すべてのパイプラインステージに存在するわけではない層だけです
    - vLLM は `nn.ModuleList` が 1 つだけ存在し、それがパイプラインステージ間に分散されることを前提とします
    - `base_model_pp_plan` が指定されない場合、Transformers モデリングバックエンドはテキストモデルの唯一の `nn.ModuleList` から分割を推定し、その周囲のパラメータを持つモジュール（入力埋め込み、最終 norm）を（宣言順に応じて）最初 / 最後のステージに、パラメータを持たないモジュール（回転位置埋め込みなど）をすべてのステージに配置します
    - `tuple` の最初の要素の `list` には、入力引数の名前が入ります
    - `tuple` の最後の要素の `list` には、モデリングのコードでその層が出力する変数の名前が入ります

### プラグイン { #plugins }

一部のモデルアーキテクチャは vLLM のプラグイン経由でサポートされています。これらのプラグインは、[プラグインシステム](../design/plugin_system.md)を通じて vLLM の機能を拡張します。

| アーキテクチャ | モデル | プラグインのリポジトリ |
| ------------ | ------ | ----------------- |
| `BartForConditionalGeneration` | BART | [bart-plugin](https://github.com/vllm-project/bart-plugin) |
| `Florence2ForConditionalGeneration` | Florence-2 | [bart-plugin](https://github.com/vllm-project/bart-plugin) |

ネイティブにサポートされていないその他のモデルアーキテクチャ、とくにエンコーダ・デコーダのモデルについては、同様のパターンに従い、プラグインシステムを通じてサポートを実装することを推奨します。

## モデルの読み込み { #loading-a-model }

### Hugging Face Hub { #hugging-face-hub }

既定では、vLLM は [Hugging Face（HF）Hub](https://huggingface.co/models) からモデルを読み込みます。モデルのダウンロード先を変更するには `HF_HOME` 環境変数を設定します。詳細は[公式ドキュメント](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)を参照してください。

あるモデルがネイティブにサポートされているかどうかは、HF リポジトリ内の `config.json` ファイルで確認できます。
`"architectures"` フィールドに以下に挙げるモデルアーキテクチャが含まれていれば、ネイティブにサポートされているはずです。

vLLM で使うために、モデルがネイティブにサポートされている_必要はありません_。
[Transformers モデリングバックエンド](#transformers)を使えば、Transformers の実装（さらには Hugging Face Model Hub 上のリモートコード）を使ってモデルを直接実行できます。

!!! tip
    実行時に本当にモデルがサポートされているかを確認する最も簡単な方法は、以下のプログラムを実行することです。

    ```python
    from vllm import LLM

    # For generative models (runner=generate) only
    llm = LLM(model=..., runner="generate")  # Name or path of your model
    output = llm.generate("Hello, my name is")
    print(output)

    # For pooling models (runner=pooling) only
    llm = LLM(model=..., runner="pooling")  # Name or path of your model
    output = llm.encode("Hello, my name is")
    print(output)
    ```

    vLLM が（生成モデルなら）テキストを、（プーリングモデルなら）hidden states を正常に返せば、そのモデルはサポートされています。

そうでない場合は、vLLM でモデルを実装する方法について[新しいモデルの追加](../contributing/model/README.md)を参照してください。
あるいは、[GitHub で issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)して vLLM でのサポートをリクエストすることもできます。

#### モデルのダウンロード { #download-a-model }

必要であれば、Hugging Face CLI を使って[モデルをダウンロード](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-download)したり、モデルリポジトリから特定のファイルを取得したりできます。

```bash
# Download a model
hf download HuggingFaceH4/zephyr-7b-beta

# Specify a custom cache directory
hf download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

# Download a specific file from a model repo
hf download HuggingFaceH4/zephyr-7b-beta eval_results.json
```

#### ダウンロード済みモデルの一覧表示 { #list-the-downloaded-models }

ローカルキャッシュに保存されたモデルを[管理](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#scan-your-cache)するには、Hugging Face CLI を使います。

```bash
# List cached models
hf cache list -q

# Show detailed (verbose) output
hf cache list

# Specify a custom cache directory
hf cache list --dir ~/.cache/huggingface/hub
```

#### キャッシュ済みモデルの削除 { #delete-a-cached-model }

キャッシュから[ダウンロード済みのモデルを削除](https://huggingface.co/docs/huggingface_hub/guides/manage-cache#clean-your-cache)するには、Hugging Face CLI を使います。

```bash
# delete all the cached objects
hf cache rm $(hf cache list -q)
```

#### プロキシを使う { #using-a-proxy }

プロキシ経由で Hugging Face からモデルを読み込む / ダウンロードする際のヒントを示します。

- セッション全体にプロキシを設定する（あるいはプロファイルのファイルに設定する）:

```shell
export http_proxy=http://your.proxy.server:port
export https_proxy=http://your.proxy.server:port
```

- 現在のコマンドにのみプロキシを設定する:

```shell
https_proxy=http://your.proxy.server:port hf download <model_name>

# or use vllm cmd directly
https_proxy=http://your.proxy.server:port  vllm serve <model_name>
```

- Python インタプリタ内でプロキシを設定する:

```python
import os

os.environ["http_proxy"] = "http://your.proxy.server:port"
os.environ["https_proxy"] = "http://your.proxy.server:port"
```

### ModelScope { #modelscope }

Hugging Face Hub の代わりに [ModelScope](https://www.modelscope.cn) のモデルを使うには、環境変数を設定します。

```shell
export VLLM_USE_MODELSCOPE=True
```

そのうえで `trust_remote_code=True` とともに使います。

```python
from vllm import LLM

llm = LLM(model=..., revision=..., runner=..., trust_remote_code=True)

# For generative models (runner=generate) only
output = llm.generate("Hello, my name is")
print(output)

# For pooling models (runner=pooling) only
output = llm.encode("Hello, my name is")
print(output)
```

## 機能の状態の凡例 { #feature-status-legend }

- ✅︎ は、そのモデルでその機能がサポートされていることを示します。

- 🚧 は、その機能が計画されているものの、そのモデルではまだサポートされていないことを示します。

- ⚠️ は、その機能は利用できるものの、既知の問題や制限がある可能性を示します。

## テキストのみの言語モデルの一覧 { #list-of-text-only-language-models }

### 生成モデル { #generative-models }

生成モデルの使い方の詳細は[このページ](generative_models.md)を参照してください。

#### テキスト生成 { #text-generation }

これらのモデルは主に [`LLM.generate`](./generative_models.md#llmgenerate) API を受け付けます。Chat / Instruct モデルはさらに [`LLM.chat`](./generative_models.md#llmchat) API もサポートします。

<style>
th {
  white-space: nowrap;
  min-width: 0 !important;
}
</style>

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `AfmoeForCausalLM` | Afmoe | TBA | ✅︎ | ✅︎ |
| `ApertusForCausalLM` | Apertus | `swiss-ai/Apertus-8B-2509`, `swiss-ai/Apertus-70B-Instruct-2509`, etc. | ✅︎ | ✅︎ |
| `ArceeForCausalLM` | Arcee (AFM) | `arcee-ai/AFM-4.5B-Base`, etc. | ✅︎ | ✅︎ |
| `ArcticForCausalLM` | Arctic | `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc. | | ✅︎ |
| `AXK1ForCausalLM` | A.X-K1 | `skt/A.X-K1`, etc. | | ✅︎ |
| `BailingMoeForCausalLM` | Ling | `inclusionAI/Ling-lite-1.5`, `inclusionAI/Ling-plus`, etc. | ✅︎ | ✅︎ |
| `BailingMoeV2ForCausalLM` | Ling | `inclusionAI/Ling-mini-2.0`, etc. | ✅︎ | ✅︎ |
| `BailingMoeV2_5ForCausalLM` | Ling | `inclusionAI/Ling-2.5-1T`, `inclusionAI/Ring-2.5-1T` | | ✅︎ |
| `BloomForCausalLM` | BLOOM, BLOOMZ, BLOOMChat | `bigscience/bloom`, `bigscience/bloomz`, etc. | | ✅︎ |
| `ChatGLMModel`, `ChatGLMForConditionalGeneration` | ChatGLM | `zai-org/chatglm2-6b`, `zai-org/chatglm3-6b`, `thu-coai/ShieldLM-6B-chatglm3`, etc. | ✅︎ | ✅︎ |
| `CohereForCausalLM`, `Cohere2ForCausalLM` | Command-R, Command-A | `CohereLabs/c4ai-command-r-v01`, `CohereLabs/c4ai-command-r7b-12-2024`, `CohereLabs/c4ai-command-a-03-2025`, `CohereLabs/command-a-reasoning-08-2025`, etc. | ✅︎ | ✅︎ |
| `Cohere2MoeForCausalLM` | North-Mini-Code | `CohereLabs/North-Mini-Code`, etc. | ✅︎ | ✅︎ |
| `CwmForCausalLM` | CWM | `facebook/cwm`, etc. | ✅︎ | ✅︎ |
| `DbrxForCausalLM` | DBRX | `databricks/dbrx-base`, `databricks/dbrx-instruct`, etc. | | ✅︎ |
| `DeciLMForCausalLM` | DeciLM | `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, etc. | ✅︎ | ✅︎ |
| `DeepseekForCausalLM` | DeepSeek | `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat`, etc. | ✅︎ | ✅︎ |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 | `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`, etc. | ✅︎ | ✅︎ |
| `DeepseekV32ForCausalLM` | DeepSeek-V3.2 | `deepseek-ai/DeepSeek-V3.2`, etc. | ✅︎ | ✅︎ |
| `DeepseekV4ForCausalLM` | DeepSeek-V4 | `deepseek-ai/DeepSeek-V4-Flash`, `deepseek-ai/DeepSeek-V4-Pro`, etc. | | ✅︎ |
| `DotsOCRForCausalLM` | dots_ocr | `rednote-hilab/dots.ocr` | ✅︎ | ✅︎ |
| `Ernie4_5ForCausalLM` | Ernie4.5 | `baidu/ERNIE-4.5-0.3B-PT`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5_MoeForCausalLM` | Ernie4.5MoE | `baidu/ERNIE-4.5-21B-A3B-PT`, `baidu/ERNIE-4.5-300B-A47B-PT`, etc. | ✅︎ | ✅︎ |
| `ExaoneForCausalLM` | EXAONE-3 | `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc. | ✅︎ | ✅︎ |
| `ExaoneMoEForCausalLM` | K-EXAONE | `LGAI-EXAONE/K-EXAONE-236B-A23B`, etc. | | |
| `Exaone4ForCausalLM` | EXAONE-4 | `LGAI-EXAONE/EXAONE-4.0-32B`, etc. | ✅︎ | ✅︎ |
| `Fairseq2LlamaForCausalLM` | Llama (fairseq2 format) | `mgleize/fairseq2-dummy-Llama-3.2-1B`, etc. | ✅︎ | ✅︎ |
| `FalconForCausalLM` | Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc. | | ✅︎ |
| `FalconMambaForCausalLM` | FalconMamba | `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct`, etc. | | ✅︎ |
| `FalconH1ForCausalLM` | Falcon-H1 | `tiiuae/Falcon-H1-34B-Base`, `tiiuae/Falcon-H1-34B-Instruct`, etc. | ✅︎ | ✅︎ |
| `FlexOlmoForCausalLM` | FlexOlmo | `allenai/FlexOlmo-7x7B-1T`, `allenai/FlexOlmo-7x7B-1T-RT`, etc. | | ✅︎ |
| `GemmaForCausalLM` | Gemma | `google/gemma-2b`, `google/gemma-1.1-2b-it`, etc. | ✅︎ | ✅︎ |
| `Gemma2ForCausalLM` | Gemma 2 | `google/gemma-2-9b`, `google/gemma-2-27b`, etc. | ✅︎ | ✅︎ |
| `Gemma3ForCausalLM` | Gemma 3 | `google/gemma-3-1b-it`, etc. | ✅︎ | ✅︎ |
| `Gemma3nForCausalLM` | Gemma 3n | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
| `Gemma4ForCausalLM` | Gemma 4 | `google/gemma-4-E2B-it`, etc. | ✅︎ | ✅︎ |
| `GlmForCausalLM` | GLM-4 | `zai-org/glm-4-9b-chat-hf`, etc. | ✅︎ | ✅︎ |
| `Glm4ForCausalLM` | GLM-4-0414 | `zai-org/GLM-4-32B-0414`, etc. | ✅︎ | ✅︎ |
| `Glm4MoeForCausalLM` | GLM-4.5, GLM-4.6, GLM-4.7 | `zai-org/GLM-4.5`, etc. | ✅︎ | ✅︎ |
| `Glm4MoeLiteForCausalLM` | GLM-4.7-Flash | `zai-org/GLM-4.7-Flash`, etc. | ✅︎ | ✅︎ |
| `GlmMoeDsaForCausalLM` | GLM-5, GLM-5.1, GLM-5.2 | `zai-org/GLM-5`, etc. | ✅︎ | ✅︎ |
| `GPT2LMHeadModel` | GPT-2 | `openai-community/gpt2`, `openai-community/gpt2-xl`, etc. | | ✅︎ |
| `GPTJForCausalLM` | GPT-J | `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc. | | ✅︎ |
| `GPTNeoXForCausalLM` | GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc. | | ✅︎ |
| `GptOssForCausalLM` | GPT-OSS | `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | ✅︎ | ✅︎ |
| `GraniteForCausalLM` | Granite 3.0, Granite 3.1, PowerLM | `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b`, etc. | ✅︎ | ✅︎ |
| `GraniteMoeForCausalLM` | Granite 3.0 MoE, PowerMoE | `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b`, etc. | ✅︎ | ✅︎ |
| `GraniteMoeHybridForCausalLM` | Granite 4.0 MoE Hybrid | `ibm-granite/granite-4.0-tiny-preview`, etc. | ✅︎ | ✅︎ |
| `GraniteMoeSharedForCausalLM` | Granite MoE Shared | `ibm-research/moe-7b-1b-active-shared-experts` (test model) | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
| `HrmTextForCausalLM` | HRM-Text | `sapientinc/HRM-Text-1B`, etc. | | |
| `HunYuanDenseV1ForCausalLM` | Hunyuan Dense | `tencent/Hunyuan-7B-Instruct` | ✅︎ | ✅︎ |
| `HunYuanMoEV1ForCausalLM` | Hunyuan-A13B | `tencent/Hunyuan-A13B-Instruct`, `tencent/Hunyuan-A13B-Pretrain`, `tencent/Hunyuan-A13B-Instruct-FP8`, etc. | ✅︎ | ✅︎ |
| `HYV3ForCausalLM` | HY3 | `tencent/Hy3-preview-Base`, `tencent/Hy3-preview` | ✅︎ | ✅︎ |
| `HyperCLOVAXForCausalLM` | HyperCLOVAX-SEED-Think-14B | `naver-hyperclovax/HyperCLOVAX-SEED-Think-14B` | ✅︎ | ✅︎ |
| `InternLM2ForCausalLM` | InternLM2 | `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc. | ✅︎ | ✅︎ |
| `InternLM3ForCausalLM` | InternLM3 | `internlm/internlm3-8b-instruct`, etc. | ✅︎ | ✅︎ |
| `IQuestCoderForCausalLM` | IQuestCoderV1 | `IQuestLab/IQuest-Coder-V1-40B-Instruct`, etc. | | |
| `IQuestLoopCoderForCausalLM` | IQuestLoopCoderV1 | `IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct`, etc. | | |
| `Jais2ForCausalLM` | Jais2 | `inceptionai/Jais-2-8B-Chat`, `inceptionai/Jais-2-70B-Chat`, etc. | | ✅︎ |
| `JambaForCausalLM` | Jamba | `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1`, etc. | ✅︎ | ✅︎ |
| `KimiLinearForCausalLM` | Kimi-Linear-48B-A3B-Base, Kimi-Linear-48B-A3B-Instruct | `moonshotai/Kimi-Linear-48B-A3B-Base`, `moonshotai/Kimi-Linear-48B-A3B-Instruct` | | ✅︎ |
| `Lfm2ForCausalLM` | LFM2 | `LiquidAI/LFM2-1.2B`, `LiquidAI/LFM2-700M`, `LiquidAI/LFM2-350M`, etc. | ✅︎ | ✅︎ |
| `Lfm2MoeForCausalLM` | LFM2MoE | `LiquidAI/LFM2-8B-A1B-preview`, etc. | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama 3.1, Llama 3, Llama 2, LLaMA, Yi | `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc. | ✅︎ | ✅︎ |
| `LongcatFlashForCausalLM` | LongCat-Flash | `meituan-longcat/LongCat-Flash-Chat`, `meituan-longcat/LongCat-Flash-Chat-FP8` | ✅︎ | ✅︎ |
| `MambaForCausalLM` | Mamba | `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf`, etc. | | ✅︎ |
| `Mamba2ForCausalLM` | Mamba2 | `mistralai/Mamba-Codestral-7B-v0.1`, etc. | | ✅︎ |
| `MellumForCausalLM` | Mellum 2 | `JetBrains/Mellum2-12B-A2.5B-Base`, etc. | | ✅︎ |
| `MiMoForCausalLM` | MiMo | `XiaomiMiMo/MiMo-7B-RL`, etc. | ✅︎ | ✅︎ |
| `MiMoV2FlashForCausalLM` | MiMoV2Flash | `XiaomiMiMo/MiMo-V2-Flash`, etc. | | ✅︎ |
| `MiMoV2ForCausalLM` | MiMoV2Pro | `XiaomiMiMo/MiMo-V2.5-Pro`, etc. | | ✅︎ |
| `MiniCPMForCausalLM` | MiniCPM | `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft`, etc. | ✅︎ | ✅︎ |
| `MiniCPM3ForCausalLM` | MiniCPM3 | `openbmb/MiniCPM3-4B`, etc. | ✅︎ | ✅︎ |
| `MiniMaxM2ForCausalLM` | MiniMax-M2, MiniMax-M2.1 | `MiniMaxAI/MiniMax-M2`, etc. | ✅︎ | ✅︎ |
| `MiniMaxM3SparseForCausalLM` | MiniMax-M3 | `MiniMaxAI/MiniMax-M3`, `MiniMaxAI/MiniMax-M3-MXFP8`, etc. | | ✅︎ |
| `MistralForCausalLM` | Ministral-3, Mistral, Mistral-Instruct | `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc. | ✅︎ | ✅︎ |
| `MistralLarge3ForCausalLM` | Mistral-Large-3-675B-Base-2512, Mistral-Large-3-675B-Instruct-2512 | `mistralai/Mistral-Large-3-675B-Base-2512`, `mistralai/Mistral-Large-3-675B-Instruct-2512`, etc. | ✅︎ | ✅︎ |
| `MixtralForCausalLM` | Mixtral-8x7B, Mixtral-8x7B-Instruct | `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc. | ✅︎ | ✅︎ |
| `MPTForCausalLM` | MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter | `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc. | | ✅︎ |
| `NemotronForCausalLM` | Nemotron-3, Nemotron-4, Minitron | `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8`, etc. | ✅︎ | ✅︎ |
| `NemotronHForCausalLM` | Nemotron-H | `nvidia/Nemotron-H-8B-Base-8K`, `nvidia/Nemotron-H-47B-Base-8K`, `nvidia/Nemotron-H-56B-Base-8K`, etc. | ✅︎ | ✅︎ |
| `Olmo3ForCausalLM` | OLMo3 | `allenai/Olmo-3-7B-Instruct`, `allenai/Olmo-3-32B-Think`, etc. | ✅︎ | ✅︎ |
| `OlmoHybridForCausalLM` | OLMo Hybrid | `allenai/Olmo-Hybrid-7B` | ✅︎ | ✅︎ |
| `OlmoeForCausalLM` | OLMoE | `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct`, etc. | | ✅︎ |
| `OPTForCausalLM` | OPT, OPT-IML | `facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc. | ✅︎ | ✅︎ |
| `OrionForCausalLM` | Orion | `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc. | | ✅︎ |
| `OuroForCausalLM` | ouro | `ByteDance/Ouro-1.4B`, `ByteDance/Ouro-2.6B`, etc. | ✅︎ | |
| `PanguEmbeddedForCausalLM` | openPangu-Embedded-7B | `FreedomIntelligence/openPangu-Embedded-7B-V1.1` | ✅︎ | ✅︎ |
| `PanguProMoEV2ForCausalLM` | openpangu-pro-moe-v2 | | ✅︎ | ✅︎ |
| `PanguUltraMoEForCausalLM` | openpangu-ultra-moe-718b-model | `FreedomIntelligence/openPangu-Ultra-MoE-718B-V1.1` | ✅︎ | ✅︎ |
| `Param2MoEForCausalLM` | param2moe | `bharatgenai/Param2-17B-A2.4B-Thinking`, etc. | ✅︎ | ✅︎ |
| `PhiForCausalLM` | Phi | `microsoft/phi-1_5`, `microsoft/phi-2`, etc. | ✅︎ | ✅︎ |
| `Phi3ForCausalLM` | Phi-4, Phi-3 | `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct`, etc. | ✅︎ | ✅︎ |
| `PhiMoEForCausalLM` | Phi-3.5-MoE | `microsoft/Phi-3.5-MoE-instruct`, etc. | ✅︎ | ✅︎ |
| `Plamo2ForCausalLM` | PLaMo2 | `pfnet/plamo-2-1b`, `pfnet/plamo-2-8b`, etc. | ✅ | ✅︎ |
| `Plamo3ForCausalLM` | PLaMo3 | `pfnet/plamo-3-nict-2b-base`, `pfnet/plamo-3-nict-8b-base`, etc. | ✅ | ✅︎ |
| `Qwen2ForCausalLM` | QwQ, Qwen2 | `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc. | ✅︎ | ✅︎ |
| `Qwen2MoeForCausalLM` | Qwen2MoE | `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc. | ✅︎ | ✅︎ |
| `Qwen3ForCausalLM` | Qwen3 | `Qwen/Qwen3-8B`, etc. | ✅︎ | ✅︎ |
| `Qwen3MoeForCausalLM` | Qwen3MoE | `Qwen/Qwen3-30B-A3B`, etc. | ✅︎ | ✅︎ |
| `Qwen3NextForCausalLM` | Qwen3NextMoE | `Qwen/Qwen3-Next-80B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
| `RWForCausalLM` | Falcon RW | `tiiuae/falcon-40b`, etc. | | ✅︎ |
| `Rnj1ForCausalLM` | Rnj1 | `EssentialAI/rnj-1-instruct`, etc. | | |
| `SarvamMoEForCausalLM` | Sarvam 2 | `sarvamai/sarvam2-30b-a3b`, etc. | ✅︎ | ✅︎ |
| `SarvamMLAForCausalLM` | Sarvam 2 | `sarvamai/sarvam2-105b-a9b`, etc. | | ✅︎ |
| `SeedOssForCausalLM` | SeedOss | `ByteDance-Seed/Seed-OSS-36B-Instruct`, etc. | ✅︎ | ✅︎ |
| `SolarForCausalLM` | Solar Pro | `upstage/solar-pro-preview-instruct`, etc. | ✅︎ | ✅︎ |
| `StableLmForCausalLM` | StableLM | `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc. | | |
| `StableLMEpochForCausalLM` | StableLM Epoch | `stabilityai/stablelm-zephyr-3b`, etc. | | ✅︎ |
| `Step1ForCausalLM` | Step-Audio | `stepfun-ai/Step-Audio-EditX`, etc. | ✅︎ | ✅︎ |
| `Step3p5ForCausalLM` | Step-3.5-flash | `stepfun-ai/Step-3.5-Flash`, etc. | | ✅︎ |
| `TeleChat2ForCausalLM` | TeleChat2 | `Tele-AI/TeleChat2-3B`, `Tele-AI/TeleChat2-7B`, `Tele-AI/TeleChat2-35B`, etc. | ✅︎ | ✅︎ |
| `TeleChat3ForCausalLM` | TeleChat3 | `Tele-AI/TeleChat3-36B-Thinking`, `Tele-AI/TeleChat3-Coder-36B-Thinking`, etc. | ✅︎ | ✅︎ |
| `TeleFLMForCausalLM` | TeleFLM | `CofeAI/FLM-2-52B-Instruct-2407`, `CofeAI/Tele-FLM`, etc. | ✅︎ | ✅︎ |
| `Zamba2ForCausalLM` | Zamba2 | `Zyphra/Zamba2-7B-instruct`, `Zyphra/Zamba2-2.7B-instruct`, `Zyphra/Zamba2-1.2B-instruct`, etc. | | |

一部のモデルは [Transformers モデリングバックエンド](#transformers)経由でのみサポートされています。以下の表は、この形で公式にサポートしているモデルを示すものです。ログには Transformers モデリングバックエンドが使われている旨が出力されますが、これがフォールバック動作であるという警告は表示されません。つまり、以下に挙げたモデルで問題が発生した場合は、[Issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)していただければ、可能な限り修正に努めます。

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `GPTBigCodeForCausalLM` | StarCoder, SantaCoder, WizardCoder | `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc. | ✅︎ | |
| `OlmoForCausalLM` | OLMo | `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc. | ✅︎ | ✅︎ |
| `Olmo2ForCausalLM` | OLMo2 | `allenai/OLMo-2-0425-1B`, etc. | ✅︎ | ✅︎ |
| `SmolLM3ForCausalLM` | SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | ✅︎ | ✅︎ |
| `Starcoder2ForCausalLM` | Starcoder2 | `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc. | ✅︎ | ✅︎ |

!!! note
    現時点で、ROCm 版の vLLM は Mistral と Mixtral について最大 4096 のコンテキスト長のみをサポートしています。

## マルチモーダル言語モデルの一覧 { #list-of-multimodal-language-models }

モデルに応じて、以下のモダリティがサポートされます。

- **T**ext（テキスト）
- **I**mage（画像）
- **V**ideo（動画）
- **A**udio（音声）

`+` で結ばれたモダリティは、任意の組み合わせがサポートされます。

- 例: `T + I` は、テキストのみ、画像のみ、テキストと画像の組み合わせの入力をサポートすることを意味します。

一方、`/` で区切られたモダリティは排他的です。

- 例: `T / I` は、テキストのみと画像のみの入力はサポートするが、テキストと画像を組み合わせた入力はサポートしないことを意味します。

マルチモーダル入力をモデルに渡す方法については[このページ](../features/multimodal_inputs.md)を参照してください。

!!! tip
    Llama-4、Step3、Mistral-3、Qwen-3.5 のようなハイブリッド専用モデルでは、サポートされるマルチモーダルモダリティをすべて 0 に設定する（`--language-model-only`）ことでテキストのみのモードを有効にできます。これによりマルチモーダルモジュールが読み込まれなくなり、KV キャッシュ用に GPU メモリを多く確保できます。

!!! note
    vLLM は現時点で、ほとんどのマルチモーダルモデルについて言語バックボーンへの LoRA アダプタの追加をサポートしています。さらに、一部のマルチモーダルモデルについては tower モジュールや connector モジュールへの LoRA 追加を実験的にサポートしています。[このページ](../features/lora.md)を参照してください。

### 生成モデル { #generative-models_1 }

生成モデルの使い方の詳細については[このページ](generative_models.md)を参照してください。

#### テキスト生成 { #text-generation_1 }

これらのモデルは主に [`LLM.generate`](./generative_models.md#llmgenerate) API を受け付けます。Chat / Instruct モデルはさらに [`LLM.chat`](./generative_models.md#llmchat) API もサポートします。

| アーキテクチャ | モデル | 入力 | HF モデルの例 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | -------------------- | ------------------------- |
| `AriaForConditionalGeneration` | Aria | T + I<sup>+</sup> | `rhymes-ai/Aria` | | |
| `AudioFlamingo3ForConditionalGeneration` | AudioFlamingo3 | T + A | `nvidia/audio-flamingo-3-hf`, `nvidia/music-flamingo-hf` | ✅︎ | ✅︎ |
| `BagelForConditionalGeneration` | BAGEL | T + I<sup>+</sup> | `ByteDance-Seed/BAGEL-7B-MoT` | ✅︎ | ✅︎ |
| `BeeForConditionalGeneration` | Bee-8B | T + I<sup>E+</sup> | `Open-Bee/Bee-8B-RL`, `Open-Bee/Bee-8B-SFT` | | ✅︎ |
| `Blip2ForConditionalGeneration` | BLIP-2 | T + I<sup>E</sup> | `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b`, etc. | ✅︎ | ✅︎ |
| `ChameleonForConditionalGeneration` | Chameleon | T + I | `facebook/chameleon-7b`, etc. | | ✅︎ |
| `CheersForConditionalGeneration` | Cheers | T + I | `ai9stars/Cheers` | | ✅︎ |
| `Cohere2VisionForConditionalGeneration` | Command A Vision, Command-A+ | T + I<sup>+</sup> | `CohereLabs/command-a-vision-07-2025`, `CohereLabs/command-a-plus-05-2026`, etc. | | ✅︎ |
| `Cosmos3ForConditionalGeneration` | Cosmos3 (understanding tower) | T + I<sup>E+</sup> + V<sup>E+</sup> | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` | | ✅︎ |
| `Cosmos3EdgeForConditionalGeneration` | Cosmos3-Edge (understanding tower) | T + I<sup>E+</sup> + V<sup>E+</sup> | `nvidia/Cosmos3-Edge` | | ✅︎ |
| `DeepseekVLV2ForCausalLM` | DeepSeek-VL2 | T + I<sup>+</sup> | `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`, etc. | | ✅︎ |
| `DeepseekOCRForCausalLM` | DeepSeek-OCR | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR`, etc. | ✅︎ | ✅︎ |
| `DeepseekOCR2ForCausalLM` | DeepSeek-OCR-2 | T + I<sup>+</sup> | `deepseek-ai/DeepSeek-OCR-2`, etc. | ✅︎ | ✅︎ |
| `Eagle2_5_VLForConditionalGeneration` | Eagle2.5-VL | T + I<sup>E+</sup> | `nvidia/Eagle2.5-8B`, etc. | ✅︎ | ✅︎ |
| `Ernie4_5_VLMoeForConditionalGeneration` | Ernie4.5-VL | T + I<sup>+</sup>/ V<sup>+</sup> | `baidu/ERNIE-4.5-VL-28B-A3B-PT`, `baidu/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
| `Exaone4_5_ForConditionalGeneration` | EXAONE-4.5 | T + I<sup>E+</sup> | `LGAI-EXAONE/EXAONE-4.5-33B`, etc. | ✅︎ | ✅︎ |
| `Gemma3ForConditionalGeneration` | Gemma 3 | T + I<sup>E+</sup> | `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, etc. | ✅︎ | ✅︎ |
| `Gemma3nForConditionalGeneration` | Gemma 3n | T + I + A | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
| `Gemma4ForConditionalGeneration` | Gemma 4 | T + I<sup>+</sup> + V + A<sup>*</sup> | `google/gemma-4-E2B-it`, etc. | | ✅︎ |
| `Gemma4UnifiedForConditionalGeneration` | Gemma 4 Unified | T + I<sup>+</sup> + V + A | `google/gemma-4-12B-it`, etc. | | ✅︎ |
| `GLM4VForCausalLM`<sup>^</sup> | GLM-4V | T + I | `zai-org/glm-4v-9b`, `zai-org/cogagent-9b-20241220`, etc. | ✅︎ | ✅︎ |
| `Glm4vForConditionalGeneration` | GLM-4.1V-Thinking | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.1V-9B-Thinking`, etc. | ✅︎ | ✅︎ |
| `Glm4vMoeForConditionalGeneration` | GLM-4.5V | T + I<sup>E+</sup> + V<sup>E+</sup> | `zai-org/GLM-4.5V`, etc. | ✅︎ | ✅︎ |
| `GlmOcrForConditionalGeneration` | GLM-OCR | T + I<sup>E+</sup> | `zai-org/GLM-OCR`, etc. | ✅︎ | ✅︎ |
| `Granite4VisionForConditionalGeneration` | Granite 4 Vision | T + I<sup>E+</sup> | `ibm-granite/granite-4.1-3b-vision`, etc. | ✅︎ | ✅︎ |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | T + A | `ibm-granite/granite-speech-3.3-8b` | ✅︎ | ✅︎ |
| `GraniteSpeechPlusForConditionalGeneration` | Granite Speech Plus | T + A | `ibm-granite/granite-speech-4.1-2b-plus` | ✅︎ | ✅︎ |
| `HCXVisionForCausalLM` | HyperCLOVAX-SEED-Vision-Instruct-3B | T + I<sup>+</sup> + V<sup>+</sup> | `naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B` | | |
| `HCXVisionV2ForCausalLM` | HyperCLOVAX-SEED-Think-32B | T + I<sup>+</sup> + V<sup>+</sup> | `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B` | | |
| `H2OVLChatModel` | H2OVL | T + I<sup>E+</sup> | `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b`, etc. | ✅︎ | ✅︎ |
| `HunYuanVLForConditionalGeneration` | HunyuanOCR | T + I<sup>E+</sup> | `tencent/HunyuanOCR`, etc. | ✅︎ | ✅︎ |
| `Idefics3ForConditionalGeneration` | Idefics3 | T + I | `HuggingFaceM4/Idefics3-8B-Llama3`, etc. | ✅︎ | |
| `IsaacForConditionalGeneration` | Isaac | T + I<sup>+</sup> | `PerceptronAI/Isaac-0.1` | ✅︎ | ✅︎ |
| `InternS1ForConditionalGeneration` | Intern-S1 | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S1`, `internlm/Intern-S1-mini`, etc. | ✅︎ | ✅︎ |
| `InternS1ProForConditionalGeneration` | Intern-S1-Pro | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S1-Pro`, etc. | ✅︎ | ✅︎ |
| `InternS2PreviewForConditionalGeneration` | Intern-S2-Preview | T + I<sup>E+</sup> + V<sup>E+</sup> | `internlm/Intern-S2-Preview`, etc. | ✅︎ | ✅︎ |
| `InternVLChatModel` | InternVL 3.5, InternVL 3.0, InternVideo 2.5, InternVL 2.5, InternVL 2.0 | T + I<sup>E+</sup> + (V<sup>E+</sup>) | `OpenGVLab/InternVL3_5-14B`, `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/InternVL2-4B`, etc. | ✅︎ | ✅︎ |
| `InternVLForConditionalGeneration` | InternVL 3.0 (HF format) | T + I<sup>E+</sup> + V<sup>E+</sup> | `OpenGVLab/InternVL3-1B-hf`, etc. | ✅︎ | ✅︎ |
| `KananaVForConditionalGeneration` | Kanana-V | T + I<sup>+</sup> | `kakaocorp/kanana-1.5-v-3b-instruct`, etc. | | ✅︎ |
| `KeyeForConditionalGeneration` | Keye-VL-8B-Preview | T + I<sup>E+</sup> + V<sup>E+</sup> | `Kwai-Keye/Keye-VL-8B-Preview` | ✅︎ | ✅︎ |
| `KeyeVL1_5ForConditionalGeneration` | Keye-VL-1_5-8B | T + I<sup>E+</sup> + V<sup>E+</sup> | `Kwai-Keye/Keye-VL-1_5-8B` | ✅︎ | ✅︎ |
| `KimiAudioForConditionalGeneration` | Kimi-Audio | T + A<sup>+</sup> | `moonshotai/Kimi-Audio-7B-Instruct` | | ✅︎ |
| `KimiK25ForConditionalGeneration` | Kimi-K2.5 | T + I<sup>+</sup> | `moonshotai/Kimi-K2.5` | | ✅︎ |
| `KimiVLForConditionalGeneration` | Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking | T + I<sup>+</sup> | `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking` | | ✅︎ |
| `LightOnOCRForConditionalGeneration` | LightOnOCR-1B | T + I<sup>+</sup> | `lightonai/LightOnOCR-1B`, etc | ✅︎ | ✅︎ |
| `Lfm2VlForConditionalGeneration` | LFM2-VL | T + I<sup>+</sup> | `LiquidAI/LFM2-VL-450M`, `LiquidAI/LFM2-VL-3B`, `LiquidAI/LFM2-VL-8B-A1B`, etc. | ✅︎ | ✅︎ |
| `Llama4ForConditionalGeneration` | Llama 4 | T + I<sup>+</sup> | `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, etc. | ✅︎ | ✅︎ |
| `Llama_Nemotron_Nano_VL` | Llama Nemotron Nano VL | T + I<sup>E+</sup> | `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` | ✅︎ | ✅︎ |
| `LlavaForConditionalGeneration` | LLaVA-1.5, Pixtral (HF Transformers) | T + I<sup>E+</sup> | `llava-hf/llava-1.5-7b-hf`, `mistral-community/pixtral-12b`, etc. | ✅︎ | ✅︎ |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT, Granite Vision | T + I<sup>E+</sup> | `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, `ibm-granite/granite-vision-3.3-2b`, etc. | | ✅︎ |
| `LlavaNextVideoForConditionalGeneration` | LLaVA-NeXT-Video | T + V | `llava-hf/LLaVA-NeXT-Video-7B-hf`, etc. | ✅︎ | ✅︎ |
| `LlavaOnevision2ForConditionalGeneration` | LLaVA-OneVision-2 | T + I<sup>+</sup> + V<sup>+</sup> | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | | |
| `LlavaOnevisionForConditionalGeneration` | LLaVA-Onevision | T + I<sup>+</sup> + V<sup>+</sup> | `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc. | | ✅︎ |
| `MiDashengLMModel` | MiDashengLM | T + A<sup>+</sup> | `mispeech/midashenglm-7b` | | ✅︎ |
| `MiMoV2OmniForCausalLM` | MiMo-V2.5-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `XiaomiMiMo/MiMo-V2.5-Omni` | | ✅︎ |
| `MiniCPMO` | MiniCPM-O | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>E+</sup> | `openbmb/MiniCPM-o-2_6`, etc. | ✅︎ | ✅︎ |
| `MiniCPMV` | MiniCPM-V | T + I<sup>E+</sup> + V<sup>E+</sup> | `openbmb/MiniCPM-V-2` (see note), `openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6`, `openbmb/MiniCPM-V-4`, `openbmb/MiniCPM-V-4_5`, `openbmb/MiniCPM-V-4_6`, etc. | ✅︎ | |
| `MiniMaxM3SparseForConditionalGeneration` | MiniMax-M3 | T + I<sup>+</sup> + V<sup>+</sup> | `MiniMaxAI/MiniMax-M3`, `MiniMaxAI/MiniMax-M3-MXFP8`, etc. | | ✅︎ |
| `MiniMaxVL01ForConditionalGeneration` | MiniMax-VL | T + I<sup>E+</sup> | `MiniMaxAI/MiniMax-VL-01`, etc. | | ✅︎ |
| `Mistral3ForConditionalGeneration` | Mistral3 (HF Transformers) | T + I<sup>+</sup> | `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, etc. | ✅︎ | ✅︎ |
| `MolmoForCausalLM` | Molmo | T + I<sup>+</sup> | `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924`, etc. | ✅︎ | ✅︎ |
| `Molmo2ForConditionalGeneration` | Molmo2 | T + I<sup>+</sup> / V | `allenai/Molmo2-4B`, `allenai/Molmo2-8B`, `allenai/Molmo2-O-7B`, `allenai/MolmoWeb-4B`<sup>^</sup>, `allenai/MolmoWeb-8B`<sup>^</sup> | ✅︎ | ✅︎ |
| `MossAudioModel` | MOSS-Audio | T + A<sup>+</sup> | `OpenMOSS-Team/MOSS-Audio-4B-Instruct`, `OpenMOSS-Team/MOSS-Audio-4B-Thinking`, `OpenMOSS-Team/MOSS-Audio-8B-Instruct`, `OpenMOSS-Team/MOSS-Audio-8B-Thinking` | ✅︎ | ✅︎ |
| `MossTranscribeDiarizeForConditionalGeneration` | MOSS-Transcribe-Diarize | T + A | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | | ✅︎ |
| `Moondream3ForCausalLM` | Moondream3 | T + I | `moondream/moondream3-preview` | | ✅︎ |
| `NVLM_D_Model` | NVLM-D 1.0 | T + I<sup>+</sup> | `nvidia/NVLM-D-72B`, etc. | | ✅︎ |
| `OpenCUAForConditionalGeneration` | OpenCUA-7B | T + I<sup>E+</sup> | `xlangai/OpenCUA-7B` | ✅︎ | ✅︎ |
| `OpenPanguVLForConditionalGeneration` | openpangu-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `FreedomIntelligence/openPangu-VL-7B` | ✅︎ | ✅︎ |
| `OpenVLAForActionPrediction` | OpenVLA | T + I | `openvla/openvla-7b` | | ✅︎ |
| `Ovis` | Ovis2, Ovis1.6 | T + I<sup>+</sup> | `AIDC-AI/Ovis2-1B`, `AIDC-AI/Ovis1.6-Llama3.2-3B`, etc. | | ✅︎ |
| `Ovis2_5` | Ovis2.5 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.5-9B`, etc. | | |
| `Ovis2_6ForCausalLM` | Ovis2.6 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.6-2B`, etc. | | |
| `Ovis2_6_MoeForCausalLM` | Ovis2.6 | T + I<sup>+</sup> + V | `AIDC-AI/Ovis2.6-30B-A3B`, etc. | | |
| `PaddleOCRVLForConditionalGeneration` | Paddle-OCR | T + I<sup>+</sup> | `PaddlePaddle/PaddleOCR-VL`, etc. | | |
| `PaliGemmaForConditionalGeneration` | PaliGemma, PaliGemma 2 | T + I<sup>E</sup> | `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448`, etc. | ✅︎ | ✅︎ |
| `Phi3VForCausalLM` | Phi-3-Vision, Phi-3.5-Vision | T + I<sup>E+</sup> | `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct`, etc. | | ✅︎ |
| `Phi4MMForCausalLM` | Phi-4-multimodal | T + I<sup>+</sup> / T + A<sup>+</sup> / I<sup>+</sup> + A<sup>+</sup> | `microsoft/Phi-4-multimodal-instruct`, etc. | ✅︎ | ✅︎ |
| `Phi4ForCausalLMV` | Phi-4-reasoning-vision | T + I<sup>+</sup> | `microsoft/Phi-4-reasoning-vision-15B`, etc. | | ✅︎ |
| `PixtralForConditionalGeneration` | Ministral 3 (Mistral format), Mistral 3 (Mistral format), Mistral Large 3 (Mistral format), Pixtral (Mistral format) | T + I<sup>+</sup> | `mistralai/Ministral-3-3B-Instruct-2512`, `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, `mistralai/Mistral-Large-3-675B-Instruct-2512` `mistralai/Pixtral-12B-2409` etc. | ✅︎ | ✅︎ |
| `QianfanOCRForConditionalGeneration` | QianfanOCR | T + I<sup>E+</sup> | `baidu/Qianfan-OCR`, etc. | ✅︎ | ✅︎ |
| `Qwen2AudioForConditionalGeneration` | Qwen2-Audio | T + A<sup>+</sup> | `Qwen/Qwen2-Audio-7B-Instruct` | | ✅︎ |
| `Qwen2VLForConditionalGeneration` <sup>Q</sup> | QVQ, Qwen2-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2_5_VLForConditionalGeneration` <sup>Q</sup> | Qwen2.5-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2_5OmniThinkerForConditionalGeneration` | Qwen2.5-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen2.5-Omni-3B`, `Qwen/Qwen2.5-Omni-7B` | ✅︎ | ✅︎ |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5 | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3.5-9B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3.5-35B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3VLForConditionalGeneration` <sup>Q</sup> | Qwen3-VL | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-4B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3VLMoeForConditionalGeneration` <sup>Q</sup> | Qwen3-VL-MOE | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-30B-A3B-Instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup> | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Thinking` | ✅︎ | ✅︎ |
| `Qwen3ASRForConditionalGeneration` | Qwen3-ASR | T + A<sup>+</sup> | `Qwen/Qwen3-ASR-1.7B` | ✅︎ | ✅︎ |
| `RForConditionalGeneration` | R-VL-4B | T + I<sup>E+</sup> | `YannQi/R-4B` | | ✅︎ |
| `SkyworkR1VChatModel` | Skywork-R1V-38B | T + I | `Skywork/Skywork-R1V-38B` | | ✅︎ |
| `SmolVLMForConditionalGeneration` | SmolVLM2 | T + I | `SmolVLM2-2.2B-Instruct` | ✅︎ | |
| `Step3VLForConditionalGeneration` | Step3-VL | T + I<sup>+</sup> | `stepfun-ai/step3` | | ✅︎ |
| `StepVLForConditionalGeneration` | Step3-VL-10B | T + I<sup>+</sup> | `stepfun-ai/Step3-VL-10B` | | ✅︎ |
| `Step3p7ForConditionalGeneration` | Step-3.7-Flash | T + I<sup>+</sup> | `stepfun-ai/Step-3.7-Flash` | | ✅︎ |
| `UltravoxModel` | Ultravox | T + A<sup>E+</sup> | `fixie-ai/ultravox-v0_5-llama-3_2-1b` | ✅︎ | ✅︎ |
| `UnlimitedOCRForCausalLM` | Unlimited-OCR | T + I<sup>+</sup> | `baidu/Unlimited-OCR`, etc. | ✅︎ | ✅︎ |

一部のモデルは [Transformers モデリングバックエンド](#transformers)経由でのみサポートされています。以下の表は、この形で公式にサポートしているモデルを示すものです。ログには Transformers モデリングバックエンドが使われている旨が出力されますが、これがフォールバック動作であるという警告は表示されません。つまり、以下に挙げたモデルで問題が発生した場合は、[Issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)していただければ、可能な限り修正に努めます。

| アーキテクチャ | モデル | 入力 | HF モデルの例 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | --------------------------- | --------------------------------------- |
| `Emu3ForConditionalGeneration` | Emu3 | T + I | `BAAI/Emu3-Chat-hf` | ✅︎ | ✅︎ |

<sup>^</sup> vLLM のアーキテクチャ名に合わせるため、`--hf-overrides` でアーキテクチャ名を設定する必要があります。</br>
<sup>E</sup> このモダリティでは事前計算済みの埋め込みを入力できます。</br>
<sup>+</sup> このモダリティでは 1 つのテキストプロンプトに対して複数の項目を入力できます。
<sup>*</sup> このモダリティをサポートするのはモデルの特定のバリアントのみです（下記の注記を参照）。</br>
<sup>Q</sup> `Qwen*-VL` は公式には画像の前処理に `qwen_vl_utils` を使いますが、vLLM は `transformers` の `video_processing_qwen*` を使うため、公式の Hugging Face リポジトリの例とは結果がわずかに異なります。

!!! note
    `Gemma3nForConditionalGeneration` は KV キャッシュを共有するため V1 でのみサポートされ、MobileNet-v5 の
    vision バックボーンを利用するために `timm>=1.0.17` に依存します。

    性能は主に次の理由でまだ十分に最適化されていません。

    - 音声・視覚のいずれの MM エンコーダも `transformers.AutoModel` の実装を使っています。
    - [Google のブログ](https://developers.googleblog.com/en/introducing-gemma-3n/)で説明されている PLE キャッシュやメモリ不足時のスワップはサポートされていません。これらの機能は vLLM にとってモデル固有すぎる可能性があり、特にスワップは制約の厳しい環境に向いていると考えられます。

!!! note
    `Gemma4ForConditionalGeneration` について
    - 音声入力をサポートするのは `gemma-4-E2B` と `gemma-4-E4B` のバリアントのみです。
    - このモデルは動画を直接取り込みません。ただし vLLM の Gemma 4 実装は、動画処理を内部で行うことで動画入力をサポートします。ユーザーはメッセージ構造の中で動画をそのまま vLLM に送信でき、モデルへ渡される前にテキストと画像フレームへ変換されます。
    - 投機的デコーディング向けの Gemma 4 アシスタントのチェックポイントは、汎用のドラフトモデルによる投機的デコーディングではなく、vLLM の Gemma
      4 MTP パスを使います。[Gemma 4 アシスタントモデルの MTP の例](../features/speculative_decoding/mtp.md#gemma-4-assistant-models)を参照してください。

!!! note
    `Gemma4UnifiedForConditionalGeneration` について
    - これはエンコーダを持たない Gemma 4 のバリアント（例: `gemma-4-12B-it`）です。tower ベースの `Gemma4ForConditionalGeneration` とは異なり、**SigLIP の vision エンコーダを持たず**、**音声エンコーダも持ちません**。生のピクセルパッチは、分解された位置埋め込みを伴う Dense+LayerNorm のパイプラインによって直接 LM 空間へ射影され、生の音声波形フレームもマルチモーダル埋め込み器を通じて直接射影されます。
    - すべてのモダリティ（画像・動画・音声）がサポートされます。
    - Gemma 4 Unified のアシスタントチェックポイント（`model_type: gemma4_unified_assistant`）は、tower ベースのバリアントと同じ MTP パスを使います。[Gemma 4 アシスタントモデルの MTP の例](../features/speculative_decoding/mtp.md#gemma-4-assistant-models)を参照してください。

!!! note
    `InternVLChatModel` については、現時点で動画入力をサポートするのは Qwen2.5 のテキストバックボーンを持つ InternVL2.5（`OpenGVLab/InternVL2.5-1B` など）、InternVL3、InternVL3.5 のみです。

!!! note
    `allenai/MolmoWeb-4B` や `allenai/MolmoWeb-8B` を使うには、Molmo2 アーキテクチャでチェックポイントをサービングし、
    マルチモーダルプレフィックスの attention を無効化してください。
    `--hf-overrides '{"architectures": ["Molmo2ForConditionalGeneration"], "is_mm_prefix_lm": false}'`

!!! note
    `Moondream3ForCausalLM` は `query` と `caption` それぞれに固有のプロンプトテンプレートを使います。
    ネイティブの `detect` と `point` のスキルは独自の座標デコードを必要とするため、
    この vLLM 実装では公開していません。
    [Moondream3 のプロンプトレシピ](../features/multimodal_inputs.md#moondream3-prompt-recipes)を参照してください。

!!! note
    公式の `openbmb/MiniCPM-V-2` はまだ動作しないため、現時点ではフォーク（`HwwwH/MiniCPM-V-2`）を使う必要があります。
    詳細は <https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630> を参照してください。

#### 文字起こし { #transcription }

自動音声認識のために専用に学習された Speech2Text モデルです。

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `CohereAsrForConditionalGeneration` | Cohere-Transcribe | `CohereLabs/cohere-transcribe-03-2026` | | |
| `FireRedASR2ForConditionalGeneration` | FireRedASR2 | `allendou/FireRedASR2-LLM-vllm`, etc. | | |
| `FireRedLIDForConditionalGeneration` | FireRedLID | `PatchyTisa/FireRedLID-vllm`, etc. | | |
| `FunASRForConditionalGeneration` | FunASR | `allendou/Fun-ASR-Nano-2512-vllm`, etc. | | |
| `Gemma3nForConditionalGeneration` | Gemma3n | `google/gemma-3n-E2B-it`, `google/gemma-3n-E4B-it`, etc. | | |
| `GlmAsrForConditionalGeneration` | GLM-ASR | `zai-org/GLM-ASR-Nano-2512` | ✅︎ | ✅︎ |
| `GraniteSpeechForConditionalGeneration` | Granite Speech | `ibm-granite/granite-4.0-1b-speech`, `ibm-granite/granite-speech-3.3-2b`, etc. | ✅︎ | ✅︎ |
| `GraniteSpeechPlusForConditionalGeneration` | Granite Speech Plus | `ibm-granite/granite-speech-4.1-2b-plus` | ✅︎ | ✅︎ |
| `MossTranscribeDiarizeForConditionalGeneration` | MOSS-Transcribe-Diarize | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | | ✅︎ |
| `Qwen3ASRForConditionalGeneration` | Qwen3-ASR | `Qwen/Qwen3-ASR-1.7B`, etc. | ✅︎ | ✅︎ |
| `Qwen3OmniMoeThinkerForConditionalGeneration` | Qwen3-Omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct`, etc. | | ✅︎ |
| `VoxtralForConditionalGeneration` | Voxtral (Mistral format) | `mistralai/Voxtral-Mini-3B-2507`, `mistralai/Voxtral-Small-24B-2507`, etc. | ✅︎ | ✅︎ |
| `WhisperForConditionalGeneration` | Whisper | `openai/whisper-small`, `openai/whisper-large-v3-turbo`, etc. | | |

!!! note
    `VoxtralForConditionalGeneration` を使うには `mistral-common[audio]` のインストールが必要です。

#### リアルタイム文字起こし { #realtime-transcription }

[`/v1/realtime`](../serving/online_serving/speech_to_text.md#realtime-api)
WebSocket エンドポイント経由でストリーミング文字起こしをサポートする音声モデルです。

| アーキテクチャ | モデル | HF モデルの例 | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `VoxtralRealtimeGeneration` | Voxtral Realtime | `mistralai/Voxtral-Mini-4B-Realtime-2602` | | |
| `Qwen3ASRRealtimeGeneration` | Qwen3-ASR Realtime | `Qwen/Qwen3-ASR-0.6B` | | |

!!! note
    `VoxtralRealtimeGeneration` を使うには `mistral-common[audio]` のインストールが必要で、`--tokenizer-mode mistral` を指定してサービングする必要があります。

    `Qwen3ASRRealtimeGeneration` は `config.json` から自動検出されません。
    サービング時に `--hf-overrides '{"architectures":["Qwen3ASRRealtimeGeneration"]}'`
    を渡す必要があります。

## プーリングモデル { #pooling-models }

プーリングモデルの使い方の詳細については[このページ](pooling_models/README.md)を参照してください。

!!! important
    一部のモデルアーキテクチャは生成タスクとプーリングタスクの両方をサポートするため、
    生成モードではなくプーリングモードでモデルを使うことを確実にするには、`--runner pooling` を明示的に指定してください。

特定のプーリングタスクでサポートされるモデルの詳細については、以下のリンクを参照してください。

- [分類の使い方](pooling_models/classify.md)
- [埋め込みの使い方](pooling_models/embed.md)
- [報酬の使い方](pooling_models/reward.md)
- [トークン分類の使い方](pooling_models/token_classify.md)
- [トークン埋め込みの使い方](pooling_models/token_embed.md)
- [スコアリングの使い方](pooling_models/scoring.md)
- [個別のモデルの例](pooling_models/specific_models.md)

## モデルサポートポリシー { #model-support-policy }

vLLM では、サードパーティのモデルをエコシステムへ統合しサポートすることを推進しています。私たちのアプローチは、堅牢性の必要性と、幅広いモデルをサポートするうえでの現実的な制約とのバランスを取るよう設計されています。サードパーティモデルのサポートは次のように運用しています。

1. **コミュニティ主導のサポート**: 新しいモデルの追加については、コミュニティからの貢献を歓迎します。新しいモデルのサポート要望があった場合、コミュニティからのプルリクエスト（PR）を歓迎します。これらの貢献は、transformers などの既存実装との厳密な一致よりも、生成される出力の妥当性を主な基準として評価されます。**貢献のお願い:** モデルのベンダーから直接寄せられる PR は特に歓迎します。

2. **ベストエフォートの一貫性**: vLLM で実装されたモデルと transformers などの他フレームワークとの間で一定の一貫性を保つことを目指していますが、完全な一致が常に実現できるとは限りません。高速化手法や低精度演算の利用といった要因により差異が生じることがあります。私たちが保証するのは、実装されたモデルが機能し、妥当な結果を生成することです。

    !!! tip
        Hugging Face Transformers の `model.generate` の出力と vLLM の `llm.generate` の出力を比較する際は、前者がモデルの生成 config ファイル（[generation_config.json](https://github.com/huggingface/transformers/blob/19dabe96362803fb0a9ae7073d03533966598b17/src/transformers/generation/utils.py#L1945)）を読み込んで既定の生成パラメータを適用するのに対し、後者は関数に渡されたパラメータのみを使う点に注意してください。出力を比較するときは、すべてのサンプリングパラメータが同一であることを確認してください。

3. **問題の解決とモデルの更新**: サードパーティモデルで見つけたバグや問題は報告してください。修正案は、問題の内容と提案する解決方法の根拠を明確に説明したうえで PR として提出してください。あるモデルの修正が別のモデルに影響する場合、そうしたモデル間の依存関係の指摘と対処はコミュニティに頼っています。補足: バグ修正の PR では、元の作者に知らせてフィードバックを求めるのが良い作法です。

4. **モニタリングと更新**: 特定のモデルに関心があるユーザーは、そのモデルのコミット履歴（例: main/vllm/model_executor/models ディレクトリの変更の追跡）を監視してください。こうした能動的な取り組みにより、利用しているモデルに影響しうる更新や変更を把握しやすくなります。

5. **重点の選択**: 私たちのリソースは、主にユーザーの関心と影響が大きいモデルに向けられます。利用頻度の低いモデルには注力が及ばないことがあり、その保守と改善についてはコミュニティのより積極的な役割に頼っています。

このアプローチにより、vLLM はコア開発チームと広範なコミュニティの双方が、エコシステムでサポートされるサードパーティモデルの堅牢性と多様性に貢献する協働的な環境を育んでいます。

なお、推論エンジンである vLLM は新しいモデルを生み出すものではありません。したがって、この観点では vLLM がサポートするすべてのモデルがサードパーティモデルです。

モデルのテストには次のレベルがあります。

1. **厳密な一致**: greedy デコーディングのもとで、モデルの出力を HuggingFace Transformers ライブラリでのモデルの出力と比較します。最も厳しいテストです。このテストに合格したモデルについては [models tests](https://github.com/vllm-project/vllm/blob/main/tests/models) を参照してください。
2. **出力の妥当性**: 出力のパープレキシティを測定し、明らかな誤りがないかを確認することで、モデルの出力が妥当で一貫しているかを検査します。これはやや緩いテストです。
3. **実行時の動作**: モデルがエラーなく読み込まれ実行できるかを確認します。最も緩いテストです。このテストに合格したモデルについては[機能テスト](../../tests)と[例](../../examples)を参照してください。
4. **コミュニティからのフィードバック**: モデルについてのフィードバックはコミュニティに頼っています。モデルが壊れている、あるいは期待どおりに動作しない場合は、Issue を作成して報告するか、修正のプルリクエストを作成していただけると助かります。残りのモデルはこのカテゴリに該当します。
