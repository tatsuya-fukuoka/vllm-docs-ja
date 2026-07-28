# IndexCache { #indexcache }

IndexCache は、top-k のインデックスを層をまたいでキャッシュ・再利用することで、DeepSeek-V3.2 (DSA) のモデルにおける冗長な top-k 計算を削減します。

## 背景 { #background }

DeepSeek-V3.2 は DeepSeek Sparse Attention (DSA) の機構を採用しており、top-k のトークン選択を層ごとに計算します。層数の多い深いモデルでは、この計算のコストが無視できません。IndexCache は、前の層のインデックスを再利用することで冗長な top-k 計算を省けるようにします。

参考: [IndexCache の論文](https://arxiv.org/abs/2603.12201)（英語）

## 使い方 { #usage }

### CLI { #cli }

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ...
```

### 設定のリファレンス { #configuration-reference }

| パラメータ           | 型   | 既定値  | 説明                                                                                                                                              |
|----------------------|------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `use_index_cache`    | bool | false   | IndexCache を有効にする。この機能を使うには true にする必要があります                                                                              |
| `index_topk_freq`    | int  | 1       | top-k を計算する頻度（層単位）。1 = すべての層で計算（実質無効）、4 = 1/4 の層で計算                                                              |
| `index_topk_pattern` | str  | null    | 層ごとの F/S パターン。設定すると index_topk_freq より優先されます。各文字が 1 つの DSA 層に対応します（F = Full、S = Shared）                     |

### 設定例 { #configuration-examples }

**`index_topk_freq` を使う場合**（N 層ごとに計算）:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' ...
```

**`index_topk_pattern` を使う場合**（層ごとに明示的に制御）:

```bash
# custom pattern for 61 layers: F = compute, S = reuse
vllm serve deepseek-ai/DeepSeek-V3.2 \
    --hf-overrides '{"use_index_cache": true, "index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSF"}'
```

## 仕組み { #how-it-works }

1. IndexCache を有効にすると、`"F"`（Full）が指定された層が top-k のインデックスを計算して保存します
2. 後続の `"S"`（Shared）が指定された層は、再計算する代わりに前の層のキャッシュ済みインデックスを受け取ります
3. キャッシュされたインデックスが層のスタックを通じて渡され、全体の計算量が削減されます

## 要件 { #requirements }

- DeepSeek-V3.2 または互換の DSA モデル
- `--hf-overrides` による `use_index_cache: true` の指定
