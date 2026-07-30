# Python のマルチプロセシング { #python-multiprocessing }

## デバッグ { #debugging }

既知の問題とその解決方法については、[トラブルシューティング](../usage/troubleshooting.md#python-multiprocessing)のページを参照してください。

## はじめに { #introduction }

!!! important
    ソースコードへの参照は、この文章を書いた 2024 年 12 月時点のコードの状態に対するものです。

vLLM における Python のマルチプロセシングの利用は、次の理由で複雑になっています。

- vLLM がライブラリとして使われるため、その内部コードを制御しきれないこと
- 一部のマルチプロセシングの方式と vLLM の依存パッケージとの非互換性

このドキュメントでは、vLLM がこれらの課題にどう対処しているかを説明します。

## マルチプロセシングの方式 { #multiprocessing-methods }

[Python のマルチプロセシングの方式](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)には次のものがあります。

- `spawn` - 新しい Python プロセスを起動します。Windows と macOS での既定です。
- `fork` - `os.fork()` を使って Python インタプリタを fork します。Python 3.14 より前のバージョンでは Linux での既定です。
- `forkserver` - 要求に応じて新しいプロセスを fork するサーバープロセスを起動します。Python 3.14 以降では Linux での既定です。

### トレードオフ { #tradeoffs }

`fork` は最も高速な方式ですが、スレッドを使う依存パッケージとは互換性がありません。macOS では `fork` を使うとプロセスがクラッシュすることがあります。

`spawn` は依存パッケージとの互換性が高いものの、vLLM をライブラリとして使う場合には問題になりえます。利用側のコードが `__main__` ガード（`if __name__ == "__main__":`）を使っていない場合、vLLM が新しいプロセスを起動した際にそのコードが意図せず再実行されてしまいます。これは無限再帰などの問題につながります。

`forkserver` は、必要に応じて新しいプロセスを fork するサーバープロセスを起動します。残念ながら、vLLM をライブラリとして使う場合には `spawn` と同じ問題があります。サーバープロセスは spawn された新しいプロセスとして作られるため、`__main__` ガードで保護されていないコードが再実行されます。

`spawn` と `forkserver` のどちらでも、プロセスは `fork` のようにグローバルな状態を継承することに依存してはいけません。

## 依存パッケージとの互換性 { #compatibility-with-dependencies }

vLLM の複数の依存パッケージが、`spawn` の使用を推奨または要求しています。

- <https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing>
- <https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors>
- <https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders>

これらの依存パッケージを初期化したあとに `fork` を使うと、既知の問題が発生します。

## 現在の状況（v0） { #current-state-v0 }

環境変数 `VLLM_WORKER_MULTIPROC_METHOD` で、vLLM が使う方式を制御できます。現在の既定値は `fork` です。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L339-L342>

メインプロセスが `vllm` コマンド経由で制御されている場合は、最も広く互換性のある `spawn` が使われます。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/scripts.py#L123-L140>

`multiproc_xpu_executor` は `spawn` の使用を強制します。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/executor/multiproc_xpu_executor.py#L14-L18>

その他にも、`spawn` の使用がハードコードされている箇所があります。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/distributed/device_communicators/all_reduce_utils.py#L135>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/entrypoints/openai/api_server.py#L184>

関連する PR:

- <https://github.com/vllm-project/vllm/pull/8823>

## v1 におけるかつての状況 { #prior-state-in-v1 }

v1 のエンジンコアでマルチプロセシングを使うかどうかを制御する環境変数 `VLLM_ENABLE_V1_MULTIPROCESSING` がありました。既定では無効でした。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/envs.py#L452-L454>

有効にすると、v1 の `LLMEngine` はエンジンコアを実行するための新しいプロセスを作成しました。

- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L93-L95>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/llm_engine.py#L70-L77>
- <https://github.com/vllm-project/vllm/blob/d05f88679bedd73939251a17c3d785a354b2946c/vllm/v1/engine/core_client.py#L44-L45>

既定で無効だったのは、上で述べたすべての理由、すなわち依存パッケージとの互換性と、vLLM をライブラリとして使うコードへの配慮のためです。

### v1 で行った変更 { #changes-made-in-v1 }

Python の `multiprocessing` について、どこでもうまく動く簡単な解決策はありません。第一歩として、互換性を最大化するために「ベストエフォート」でマルチプロセシングの方式を選ぶ状態に v1 を持っていくことができます。

- 既定は `fork` にする。
- メインプロセスを自分たちが制御していると分かっている場合（`vllm` が実行された場合）は `spawn` を使う。
- `cuda` がすでに初期化されていることを検出したら `spawn` を強制し、警告を出す。`fork` が壊れることは分かっているので、これが最善です。

このシナリオでもなお壊れることが分かっているのは、vLLM を呼び出す前に `cuda` を初期化する、vLLM をライブラリとして使うコードです。出力する警告では、`__main__` ガードを追加するか、マルチプロセシングを無効にするようユーザーに案内すべきです。

この既知の失敗ケースが起きると、ユーザーには状況を説明する 2 つのメッセージが表示されます。まず、vLLM からのログメッセージです。

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

次に、Python 自体が分かりやすい説明とともに例外を送出します。

```console
RuntimeError:
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
```

## 検討した代替案 { #alternatives-considered }

### `__main__` ガードの有無を検出する { #detect-if-a-__main__-guard-is-present }

vLLM をライブラリとして使うコードに `__main__` ガードがあるかどうかを検出できれば、より良い挙動にできるのではないかという提案がありました。同じ問題に直面したライブラリ作者による [Stack Overflow の投稿](https://stackoverflow.com/questions/77220442/multiprocessing-pool-in-a-python-class-without-name-main-guard)もあります。

自分が元の `__main__` プロセスにいるのか、その後に spawn されたプロセスにいるのかを検出することは可能です。しかし、コードに `__main__` ガードがあるかどうかを検出するのは簡単ではないようです。

この選択肢は現実的でないとして見送られました。

### `forkserver` を使う { #use-forkserver }

一見すると `forkserver` はこの問題に対する良い解決策に思えます。しかしその仕組み上、vLLM をライブラリとして使う場合には `spawn` と同じ課題が生じます。

### 常に `spawn` を強制する { #force-spawn-all-the-time }

これを整理する 1 つの方法は、常に `spawn` の使用を強制し、vLLM をライブラリとして使う場合には `__main__` ガードが必須であるとドキュメントに記載することです。しかしこれは既存のコードを壊し、vLLM を使いにくくしてしまいます。これは `LLM` クラスをできるだけ簡単に使えるようにしたいという方針に反します。

この負担をユーザーに押しつけるのではなく、私たちは複雑さを引き受け、できる限りうまく動くようにします。

## 今後の課題 { #future-work }

将来的には、これらの課題を回避する別のワーカー管理のアプローチを検討したいと考えています。

1. `forkserver` に似たものを実装しつつ、プロセスマネージャーを、自分たちのサブプロセスとワーカー管理用の独自エントリポイント（`vllm-manager` プロセス）として最初に起動する形にする。

2. ニーズにより適した他のライブラリを検討する。検討候補の例:

    - <https://github.com/joblib/loky>
