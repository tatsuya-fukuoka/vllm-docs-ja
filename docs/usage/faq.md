# よくある質問 { #frequently-asked-questions }

> Q: OpenAI API を使って、1 つのポートで複数のモデルをサービングするにはどうすればよいですか？

A: OpenAI 互換サーバーで複数のモデルを同時にサービングしたい、という意味であれば、現時点ではサポートされていません。サーバーのインスタンスをモデルごとに複数起動し、その手前に受信リクエストを適切なサーバーへ振り分ける層を置いてください。

---

> Q: オフライン推論で埋め込みを作るには、どのモデルを使えばよいですか？

A: [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) や [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) を試してみてください。
その他のモデルは[こちら](../models/supported_models.md)に一覧があります。

vLLM は隠れ状態を取り出すことで、[Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) や
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) のようなテキスト生成モデルを自動的に埋め込みモデルとして扱えますが、
埋め込みタスク向けに専用に学習されたモデルには品質で劣ると考えられます。

---

> Q: vLLM では、同じプロンプトの出力が実行ごとに変わることがありますか？

A: あります。vLLM は出力トークンの対数確率（logprobs）が安定することを保証していません。logprobs のばらつきは、
Torch の演算における数値的な不安定性や、バッチが変化したときのバッチ化された Torch 演算の非決定的な挙動によって生じます。詳細は
[Numerical Accuracy のセクション](https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations)（英語）を参照してください。

vLLM では、同時に処理される他のリクエスト、バッチサイズの変化、投機的デコーディングにおけるバッチ拡張などの要因により、
同じリクエストでもバッチの組まれ方が変わることがあります。このバッチのばらつきと Torch 演算の数値的な不安定性が組み合わさることで、
各ステップの logit / logprob の値がわずかに異なる場合があります。こうした差は累積し、結果としてサンプリングされるトークンが
変わることがあります。いったん異なるトークンがサンプリングされると、その後の出力はさらに乖離していきます。

## 緩和策 { #mitigation-strategies }

- 安定性を高めてばらつきを減らすには `float32` を使用してください。ただしメモリ使用量は増えます。
- `bfloat16` を使っている場合は、`float16` に切り替えることでも改善する場合があります。
- temperature > 0 でより安定した生成を得たい場合はリクエストの seed 指定が有効ですが、精度の違いによる差異は残る可能性があります。
