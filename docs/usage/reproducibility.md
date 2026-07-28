# 再現性 { #reproducibility }

vLLM は性能を優先するため、既定では結果の再現性を保証しません。再現可能な結果を得るには次のようにします。

- オフラインモードでは、`VLLM_ENABLE_V1_MULTIPROCESSING=0` を設定してスケジューリングを決定的にするか、
  [batch invariance](../features/batch_invariance.md) を有効にして出力がスケジューリングに影響されないようにします。
- オンラインモードでは、[batch invariance](../features/batch_invariance.md) を有効にする方法のみ利用できます。

例: [examples/features/batch_invariance/reproducibility_offline.py](../../examples/features/batch_invariance/reproducibility_offline.py)

!!! warning

    `VLLM_ENABLE_V1_MULTIPROCESSING=0` を設定すると、ユーザーコード
    （[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスを構築するコード）側の乱数状態が変化します。

!!! note

    上記の設定を行った場合でも、vLLM が再現性を保証するのは同じハードウェア・同じ vLLM バージョンで
    実行したときに限られます。

## グローバルな seed の設定 { #setting-the-global-seed }

vLLM の `seed` パラメータは、各種乱数生成器の状態を制御するために使われます。

特定の seed 値を指定すると、`random`、`np.random`、`torch.manual_seed` の乱数状態がそれに応じて設定されます。

### 既定の挙動 { #default-behavior }

V1 では `seed` の既定値は `0` で、各ワーカーの乱数状態が設定されます。そのため `temperature > 0` であっても、vLLM の実行ごとに結果は一貫します。

V1 では seed を「未指定」にはできません。投機的デコーディングなどのワークフローでは、
複数のワーカーが同じ出力をサンプリングする必要があるためです。詳細は <https://github.com/vllm-project/vllm/pull/17929> を参照してください。

!!! note

    ユーザーコード（[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスを構築するコード）側の乱数状態が
    vLLM によって更新されるのは、ワーカーがユーザーコードと同じプロセスで実行される場合、
    つまり `VLLM_ENABLE_V1_MULTIPROCESSING=0` のときだけです。

    既定では `VLLM_ENABLE_V1_MULTIPROCESSING=1` なので、乱数状態に依存する後続の処理が意図せず
    決定的になってしまう心配なく vLLM を使えます。
