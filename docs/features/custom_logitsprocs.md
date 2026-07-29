# カスタムのロジットプロセッサ { #custom-logits-processors }

!!! important
    ロジットプロセッサの設計変更の一部はまだ進行中で、API は近い将来変わる可能性があります。この部分の API は近いうちに安定させたいと考えています。

「カスタム」のロジットプロセッサは vLLM の利用者が書くもので、vLLM のソースコードを変更したり再コンパイルしたりすることなく、初期化時に vLLM へ読み込まれます。組み込みのロジットプロセッサの対極にあるものです。

このドキュメントでは、カスタムのロジットプロセッサの書き方、読み込み方、使い方を説明します。

## ロジットプロセッサの背景 { #logits-processors-background }

ロジットプロセッサは次トークンの確率分布を調整するもので、通常はモデルを望ましい振る舞いへ誘導することを目的とします。

vLLM では、ロジットプロセッサはバッチ単位で動作します。あるエンジンステップにおいて、ロジットプロセッサはモデルが出力した生のロジットの `(num_requests) x (vocab_size)` テンソルを受け取ります。そのロジットプロセッサを有効にしているすべてのリクエストについて、対応するロジットテンソルの行に変換を適用し、それ以外の行は変更しません。変換後のロジットテンソルは softmax に渡されます。

## カスタムのロジットプロセッサを作る { #creating-a-custom-logits-processor }

カスタムのロジットプロセッサは `vllm.v1.sample.logits_processor.LogitsProcessor` を継承し、少なくとも次のメソッドを定義する必要があります。

* `validate_params(cls, sampling_params: SamplingParams)`:
    * ロジットプロセッサが使う `SamplingParams` の引数（とくにカスタム引数）が不正な場合に `ValueError` を送出します。
    * リクエストがエントリポイントに送られると、`validate_params()` が `SamplingParams` を検証し、不正な引数を含むリクエストを拒否します。
    * **注:** カスタムのロジットプロセッサに不正なパラメータが渡るのを防ぐため、`validate_params()` を実装することが重要です。そうしないと、不正なパラメータを含むリクエストがカスタムのロジットプロセッサで予期しない挙動を引き起こす可能性があります。

* `__init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool)`
    * `vllm_config`: エンジンの設定データ構造
    * `device`: ハードウェアアクセラレータのデバイス情報
    * `is_pin_memory`: ロジットプロセッサの実装のために pin メモリが利用できるかを示すフラグ

* `apply(self, logits: torch.Tensor) -> torch.Tensor`:
    * `(num_requests) x (vocab_size)` のロジットテンソル（`logits`）を受け取ります
    * バッチ単位でロジットプロセッサの変換を適用します
    * 変換後の `(num_requests) x (vocab_size)` のロジットテンソルを返します
    * 入力のロジットはインプレースでもアウトオブプレースでも変更できます。インプレースのほうがメモリ効率に優れます

* `is_argmax_invariant(self) -> bool`:
    * そのロジットプロセッサが argmax 不変（あるリクエストについて最大ロジット値を持つトークン ID を決して変えない）であれば `True`、argmax を変える可能性があれば `False` を返します
    * `is_argmax_invariant()` は起動時に一度だけ評価されます。`True` の場合、すべてのリクエストが貪欲サンプリングを使うステップでは、vLLM はこのロジットプロセッサの適用をスキップします

* `update_state(self, batch_update: Optional["BatchUpdate"]) -> None`:
    * 現在のエンジンステップ開始時の永続バッチの状態変化を表す `BatchUpdate` データ構造を受け取ります
    * `BatchUpdate` のメンバを使ってロジットプロセッサの内部状態を更新します
    * **注:** バッチ更新のデータ構造は `None` になることがあり、これはバッチの構成に変化がないことを示します。この場合でも、ロジットプロセッサは、追加時に保持した `output_token_ids` のリストが更新されているのを踏まえて状態を更新したい場合があります。

### vLLM エンジンが `BatchUpdate` データ構造を構築する方法 { #how-the-vllm-engine-builds-the-batchupdate-data-structure }

!!! important
    ロジットプロセッサの設計変更の一部はまだ進行中です。将来的には、ロジットプロセッサを実装する際に
    バッチ状態の変化を考慮する必要はなくなり、この節の情報は不要になる見込みです。

ロジットプロセッサの `update_state()` の実装は、モデルランナーが永続バッチの状態を更新する方法として、次のモデル（ここでは `BatchUpdate` の抽象で表現）を前提とすべきです。

1. 現在のエンジンステップで完了したリクエストのインデックスを特定する

2. 現在のステップで新たに投入されたリクエストを特定する

3. Add 操作により、完了したリクエストをできるだけ多く新しいリクエストで置き換える。置き換えるリクエストのインデックスが小さい順に処理する

4. 新規リクエストと完了リクエストの数の関係に応じて:

    1. 新規と完了の数が同じであれば、次のステップへ進む

    2. *新規リクエストのほうが完了リクエストより多い場合:* 完了リクエストを置き換えなかった残りの新規リクエストで、Add 操作によりバッチを拡張する。これらの新規リクエストには `current_max_batch_index + 1` から始まる連続したインデックスを割り当てる

    3. *新規リクエストのほうが完了リクエストより少ない場合:*

        * 新規リクエストで置き換えられなかった完了リクエストに Remove 操作を適用する。これら削除されるリクエストのインデックスは、必ず前のステップで置き換えられた完了リクエストの最大インデックスより大きくなる。Remove によりバッチは非連続な状態になることがある

        * **バッチを連続にするよう「圧縮」する:** （Remove により生じた）最も小さいインデックスの空きスロットから始め、バッチ内で現在最も大きいインデックスの非空スロットから一方向の Move を適用して空きスロットを埋める。バッチが連続になるまで、空きスロットの宛先インデックスは昇順、非空スロットの元インデックスは降順の順で一方向の Move を続ける

        * **バッチを縮小する:** バッチの圧縮の副作用として、Remove により生じた空きスロットはバッチ配列の末尾に連続したブロックとしてまとまる。したがって圧縮後は、非空スロットの数を反映するよう `BatchUpdate.batch_size` を更新する

5. 効率を高めるためにバッチを並べ替える。attention バックエンドの実装と現在のバッチの特性に応じて、バッチの並べ替えのために 0 個以上の Swap の Move 操作が適用されることがある

注意点:

* ロジットプロセッサの `update_state()` メソッドは、バッチ更新の操作を「remove、add、move」の順で処理しなければなりません

* Add 操作のインデックス引数は、*その Add が発生した時点*の、すなわち Move 操作より前のインデックスを指します
    * 例: あるリクエストがインデックス 5 で Add され、その後インデックス 3 と交換された場合、`BatchUpdate.added` の Add 操作はインデックス 3 ではなく 5 に対応づけられます
    * 言い換えると、Move 操作は Add と Remove のあとに適用されると仮定できます

* Move 操作は `BatchUpdate.moved` に現れる順序で適用されると仮定できます

* 新規 / 完了のリクエストがなく、バッチの並べ替えもない場合、ロジットプロセッサへのバッチ更新は `None` になります

### カスタムのロジットプロセッサへのカスタム引数の受け渡し { #passing-custom-argument-to-a-custom-logits-processor }

組み込みのロジットプロセッサとは異なり、カスタムのロジットプロセッサは `SamplingParams` や vLLM サーバーの REST API にハードコードされていない設定引数を必要とすることがあります。この問題を解決するため、カスタムのロジットプロセッサは vLLM の[カスタム引数](./custom_arguments.md)のサポートを活用してユーザーから設定を受け取れます（もちろん、`SamplingParams` の既存フィールドを利用するカスタムのロジットプロセッサを設計しても構いません）。

### カスタムのロジットプロセッサの実装例 { #example-custom-logits-processor-implementation }

以下の作為的な例は、`(num\_requests) \times (vocab\_size)` のロジットテンソルを受け取り、1 つのトークン（`target_token`）を除くすべてのトークンを `float(-inf)` でマスクするカスタムのロジットプロセッサを実装したものです。`target_token` を指定しないリクエストでは、このロジットプロセッサは無効になります。ロジットプロセッサが有効かどうか、そしてどのトークンをマスクせずに残すかを判断するため、各リクエストに紐づく `target_token` カスタム引数を `SamplingParams.extra_args` から調べます。

??? code "カスタムのロジットプロセッサの定義例"

    ``` python
    import torch
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import (BatchUpdate,
                                                LogitsProcessor,
                                                MoveDirectionality)

    class DummyLogitsProcessor(LogitsProcessor):
        """Fake logit processor to support unit testing and examples"""

        @classmethod
        def validate_params(cls, params: SamplingParams):
            target_token: int | None = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is not None and not isinstance(target_token, int):
                raise ValueError(f"target_token value {target_token} is not int")

        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool):
            self.req_info: dict[int, int] = {}

        def is_argmax_invariant(self) -> bool:
            """Never impacts greedy sampling"""
            return False

        def update_state(self, batch_update: BatchUpdate | None):
            if not batch_update:
                return

            # Process added requests.
            for index, params, _, _ in batch_update.added:
                assert params is not None
                self.validate_params(params)
                if params.extra_args and (target_token :=
                                        params.extra_args.get("target_token")):
                    self.req_info[index] = target_token
                else: 
                    self.req_info.pop(index, None)

            if self.req_info:
                # Process removed requests.
                for index in batch_update.removed:
                    self.req_info.pop(index, None)

                # Process moved requests, unidirectional move (a->b) and swap
                # (a<->b)
                for adx, bdx, direct in batch_update.moved:
                    a_val = self.req_info.pop(adx, None)
                    b_val = self.req_info.pop(bdx, None)
                    if a_val is not None:
                        self.req_info[bdx] = a_val
                    if direct == MoveDirectionality.SWAP and b_val is not None:
                        self.req_info[adx] = b_val

        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            if not self.req_info:
                return logits

            # Save target values before modification
            cols = torch.tensor(
                list(self.req_info.values()), dtype=torch.long, device=logits.device
            )
            rows = torch.tensor(
                list(self.req_info.keys()), dtype=torch.long, device=logits.device
            )
            values_to_keep = logits[rows, cols].clone()

            # Mask all but target tokens
            logits[rows] = float('-inf')
            logits[rows, cols] = values_to_keep

            return logits

    ```

このドキュメントの以降では、カスタムのロジットプロセッサの例として `DummyLogitsProcessor` を使います。

`DummyLogitsProcessor.update_state()` の実装は、バッチ内のリクエストを `self.req_info` 辞書で「疎な」表現として保持します。辞書にキーを持つのは、`target_token` の値を指定したリクエストだけです。`update_state()` は、永続バッチに対する Add、Remove、Move の操作に応じて、保存しているリクエストのインデックスと `target_token` の値（`self.req_info` のキーと値）を調整します。

### 既存のリクエスト単位ロジットプロセッサのラップ { #wrapping-an-existing-request-level-logits-processor }

vLLM エンジンはバッチ単位でロジットプロセッサを適用しますが、個々のリクエストに対して動作する「リクエスト単位」のロジットプロセッサ実装を vLLM で使いたい利用者もいるでしょう。とくに、そのロジットプロセッサが vLLM のバージョン 0 向けに開発されたものである場合はそうです。バージョン 0 では、次の型注釈に適合する `Callable`（[こちら](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.logits_process)で説明）であることが必要でした。

``` python
RequestLogitsProcessor = Union[

    # (output token ids, logits tensor) -> logits tensor
    Callable[[list[int], Tensor], Tensor],

    # (prompt token ids, output token ids, logits tensor) -> logits tensor
    Callable[[list[int], list[int], Tensor], Tensor],
]
```

リクエスト単位のロジットプロセッサは vLLM エンジンでは明示的に*サポートされていません*が、vLLM は既存の `Callable` なリクエスト単位ロジットプロセッサをラップし、vLLM と互換なバッチ単位のロジットプロセッサを作る便利な仕組みを提供して*います*。この `Callable` は上記の型注釈に適合している必要があります。リクエスト単位のロジットプロセッサが異なるインターフェースを持つ場合、ラップするには実装を修正するか、上記のインターフェース仕様に合わせる追加のラッパー層を実装する必要があるかもしれません。

以下の例のように `AdapterLogitsProcessor` を継承することで、リクエスト単位のロジットプロセッサをラップできます（この例では `DummyPerReqLogitsProcessor` が、ラップ対象のリクエスト単位ロジットプロセッサの代わりです）。

* リクエストのサンプリングパラメータを検証するために `AdapterLogitsProcessor.validate_params(cls,params)` をオーバーライドします。

* リクエスト単位のロジットプロセッサが最大ロジット値を持つトークンに影響し得るかどうかを正確に反映するため、`AdapterLogitsProcessor.is_argmax_invariant(self)` をオーバーライドします。

* `SamplingParams` のインスタンスから新しいリクエスト単位ロジットプロセッサのインスタンスを作るため、`AdapterLogitsProcessor.new_req_logits_processor(self,params)` をオーバーライドします。

??? code "リクエスト単位ロジットプロセッサのラップ例"

    ``` python
    ...

    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor, # Wrapper base-class
        RequestLogitsProcessor, # Request-level logitsproc type annotation
    )

    ...

    # Stand-in for your request-level logits processor:
    class DummyPerReqLogitsProcessor:
        """The request-level logits processor masks out all logits except the
        token id identified by `target_token`"""

        def __init__(self, target_token: int) -> None:
            """Specify `target_token`"""
            self.target_token = target_token

        def __call__(
            self,
            output_ids: list[int],
            logits: torch.Tensor,
        ) -> torch.Tensor:
            val_to_keep = logits[self.target_token].item()
            logits[:] = float("-inf")
            logits[self.target_token] = val_to_keep
            return logits

    ...

    # Example of wrapping the request-level logits processor:
    class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
        """Example of wrapping a fake request-level logit processor to create a
        batch-level logits processor"""

        @classmethod
        def validate_params(cls, params: SamplingParams):
            target_token: Any | None = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is not None and not isinstance(target_token, int):
                raise ValueError(
                    f"target_token value {target_token} is not int"
                )

        def is_argmax_invariant(self) -> bool:
            return False

        def new_req_logits_processor(
            self,
            params: SamplingParams,
        ) -> Optional[RequestLogitsProcessor]:
            """This method returns a new request-level logits processor, customized
            to the `target_token` value associated with a particular request.

            Returns None if the logits processor should not be applied to the
            particular request. To use the logits processor the request must have
            a "target_token" custom argument with an integer value.

            Args:
            params: per-request sampling params

            Returns:
            `Callable` request logits processor, or None
            """
            target_token: Any | None = params.extra_args and params.extra_args.get(
                "target_token"
            )
            if target_token is None:
                return None
            return DummyPerReqLogitsProcessor(target_token)
    ```

!!! note
    `new_req_logits_processor()` のオーバーライドは `None` を返すことで、そのリクエストにはラップしたロジットプロセッサを適用すべきでないことを示せます。

リクエスト単位のロジットプロセッサをラップするカスタムのサブクラス（`WrappedPerReqLogitsProcessor` など）を作ったら、次の節で説明するいずれかの方法で vLLM に渡せます。

## カスタムのロジットプロセッサを vLLM に読み込ませる方法 { #ways-to-load-your-custom-logits-processor-in-vllm }

ロジットプロセッサは初期化時に読み込まれます。重要な点として、vLLM エンジンの読み込みが完了したあとに読み込み済みのロジットプロセッサの集合を変更することはできず、個々のリクエストのために新しいロジットプロセッサをオンデマンドで読み込むこともできません。

この節では、ロジットプロセッサを vLLM から見えるようにし、vLLM に読み込ませるためのさまざまな方法を説明します。

### 方法 1: 初期化時にカスタムのロジットプロセッサの完全修飾クラス名（FQCN）を vLLM に渡す { #method-1-pass-the-custom-logits-processor-fully-qualified-class-name-fqcn-to-vllm-at-initialization-time }

この方法は、オフラインとオンラインの両方の利用シナリオでサポートされます。カスタムのロジットプロセッサの FQCN（`dotted.path.to.module:ClassName` の形式）は、Python の `LLM` および `AsyncLLM` コンストラクタの引数として、あるいは次の構文で `vllm serve` の CLI 引数として渡せます。

``` bash
vllm serve ... --logits_processors <logits processor 1> <logits processor 2> ...
```

FQCN に対する要件は次の 3 点だけです。

1. Python の `importlib.import_module()` が FQCN のドット区切りのパス部分を解決し、モジュールとして読み込めること

2. 読み込んだモジュールから FQCN のクラス名部分を import できること

3. FQCN が指すオブジェクトが `LogitsProcessor` のサブクラスであること

例を以下に示します。

??? code "Python で `LLM` にカスタムのロジットプロセッサの FQCN を渡す"

    ``` python
    # Pass in FQCN
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=["your.module.path:DummyLogitsProcessor"],
    )
    ```

??? code "Python で `AsyncLLM` にカスタムのロジットプロセッサの FQCN を渡す"

    ``` python
    # Pass in FQCN
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=["your.module.path:DummyLogitsProcessor"])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

??? code "CLI で vLLM サーバーにカスタムのロジットプロセッサの FQCN を渡す"

    ```bash
    vllm serve facebook/opt-125m --logits_processors your.module.path:DummyLogitsProcessor
    ```

### 方法 2: Python 環境にインストールされたカスタムのロジットプロセッサをエントリポイントとして自動検出する { #method-2-automatically-detect-custom-logits-processors-installed-in-your-python-environment-as-entry-points }

[`setuptools`](https://setuptools.pypa.io/en/latest/userguide/entry_point.html) を使うと、インストール済みのパッケージが「エントリポイント」と呼ばれるメタデータを通じて、他の Python プログラムのプラグインとして自分自身を提供できます。

初期化時、vLLM は `vllm.logits_processors` のエントリポイントグループを自動的に走査し、見つかったインストール済みのロジットプロセッサを読み込みます。

カスタムのロジットプロセッサを含む Python パッケージを開発したとします。そのパッケージにロジットプロセッサごとに一意なエントリポイントを追加することで、各ロジットプロセッサを vLLM に公開できます。以下の例は、プロジェクトの `pyproject.toml` にエントリポイントを追加する方法を示しています。

??? code "カスタムのロジットプロセッサを Python のエントリポイントとして公開する"

    ``` toml
    [project.entry-points."vllm.logits_processors"]
    dummy_logits_processor = "your.module.path:DummyLogitsProcessor"
    ```

パッケージをインストールすれば、vLLM の初期化時にカスタムのロジットプロセッサが自動的に読み込まれます。エントリポイントとして公開している場合、初期化時に `LLM` や `AsyncLLM` のコンストラクタ、あるいは vLLM サーバーへ明示的にカスタムのロジットプロセッサを渡す必要は*ありません*。

!!! note
    vLLM は、`vllm.logits_processors` のグループ配下のエントリポイントで公開された*すべての*ロジットプロセッサを*常に*読み込みます。

### 方法 3（オフライン限定）: Python のクラスオブジェクトを vLLM のコンストラクタに渡す { #method-3-offline-only-pass-a-python-class-object-to-the-vllm-constructor }

`LLM` および `AsyncLLM` のコンストラクタに、1 つ以上のカスタムのロジットプロセッサのクラスオブジェクトを渡せます。この方法は非常に柔軟で、ロジットプロセッサのクラスは (1) `LLM` や `AsyncLLM` をインスタンス化するのと同じ Python ソースファイル内でローカルに定義してもよく、(2) Python パッケージから import してもよいためです。

??? code "Python で `LLM` または `AsyncLLM` にカスタムのロジットプロセッサのクラスオブジェクトを渡す"

    ``` python
    # Import custom logits processor
    from some.module import DummyLogitsProcessor

    # ...or...

    # Define custom logits processor locally
    from vllm.v1.sample.logits_processor import LogitsProcessor

    class DummyLogitsProcessor(LogitsProcessor):
        # See DummyLogitsProcessor implementation above
        ...

    # Pass class object to LLM constructor
    llm = LLM(
        model="facebook/opt-125m",
        logits_processors=[DummyLogitsProcessor],
    )

    # Pass class object to AsyncLLM constructor
    engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                  logits_processors=[DummyLogitsProcessor])
    async_llm = AsyncLLM.from_engine_args(engine_args)
    ```

## リクエストに対してカスタムのロジットプロセッサを呼び出す { #invoking-a-custom-logits-processor-against-a-request }

あるリクエストについてロジットプロセッサを有効 / 無効にする必要があるか、またロジットプロセッサを設定するためにどの引数を渡す必要があるかは、カスタムのロジットプロセッサの設計によって決まります。

以下の例は、(1) 特定のリクエストでロジットプロセッサを有効にし、(2) その挙動を制御するために、利用者が `DummyLogitsProcessor` へカスタム引数（`target_token`）を渡す方法を示しています。

??? code "vLLM REST API: リクエストに対してカスタムのロジットプロセッサを設定する"

    ``` bash
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            ...
            "vllm_xargs": {"target_token": 67}
        }'
    ```

??? code "OpenAI SDK: リクエストに対してカスタムのロジットプロセッサを設定する"

    ``` python
    batch = await client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        ...,
        extra_body={
            "vllm_xargs": {
                "target_token": 67
            }
        }
    )
    ```

??? code "オフライン: `LLM` のリクエストに対してカスタムのロジットプロセッサを設定する"

    ``` python
    outputs_logitproc = llm.generate("your prompt", 
                                     SamplingParams(...,
                                        extra_args={"target_token": 67}))
    ```

??? code "オフライン: `AsyncLLM` のリクエストに対してカスタムのロジットプロセッサを設定する"

    ``` python
    async for out in engine.generate(request_id="your request id",
                                     prompt="your prompt",
                                     sampling_params=SamplingParams(...,
                                        extra_args={"target_token": 67})):

        # Process async request outputs
        ...
    ```

## カスタムのロジットプロセッサを書く際のベストプラクティス { #best-practices-for-writing-custom-logits-processors }

初期化時に vLLM がロジットプロセッサを読み込むと、以降 vLLM はエンジンステップごとにそのロジットプロセッサの `update_state()` と `apply()` を呼び出します。どちらのメソッドも、その時点で vLLM の永続バッチに存在するすべてのリクエストに対して動作します。したがって、これらのメソッドを効率的に実装することが重要です。

* ロジットプロセッサがバッチ単位で動作することを踏まえ、効率的な `apply()` と `update_state()` の実装を書いてください
    * たとえば、`apply()` の実装や `update_state()` での内部状態ベクトルの更新に、効率的なベクトル化演算を使えるかもしれません
    * ただし、そのロジットプロセッサが使われる頻度が低いと考えられる場合は、リクエスト状態を「疎な」表現で持つほうが適切なこともあります。すなわち、そのロジットプロセッサを有効にしているリクエストのメタデータだけを辞書で保持する、といった方法です
    * **注:** ラップしたリクエスト単位のロジットプロセッサでは、`apply()` と `update_state()` を実装する必要はありません。既定の `AdapterLogitsProcessor.update_state()` の実装はリクエスト状態の疎な表現を保持し、`new_req_logits_processor()` が `None` を返したリクエストは基底クラスの状態辞書に含まれません。`AdapterLogitsProcessor.apply()` の既定の実装は、入力ロジットの各行に対してリクエスト単位のロジットプロセッサを順に適用し、出力ロジットテンソルを組み立てます。この `AdapterLogitsProcessor` の既定実装の性能が不十分な場合は、リクエスト単位のロジットプロセッサをラップするのではなく、バッチ単位で動作する最適化された `apply()` と `update_state()` を持つ `LogitsProcessor` のサブクラスとして再実装してください

* 次の点はロジットプロセッサの作者が決めることです。

    1. **そのリクエストに対するロジットプロセッサの挙動を設定する、リクエストごとの属性。** カスタムのロジットプロセッサの `update_state()` のオーバーライドが、`SamplingParams` のフィールドをどうロジットプロセッサの状態へ対応づけるかを決めます

        * **注:** ラップしたリクエスト単位のロジットプロセッサでは、`new_req_logits_processor()` が、`SamplingParams` のフィールドをどう使ってリクエスト単位のロジットプロセッサのインスタンスを初期化するかを決めます。

    2. **リクエストごとにロジットプロセッサを有効 / 無効にする条件。** カスタムのロジットプロセッサを常にすべてのリクエストに作用させるつもりでない限り、あるリクエストについてロジットプロセッサを無効にできるように書くべきです。たとえば引数の既定値を `None` にする、あるいは何もしないことを表す特定の値（`0.0` など）を渡す、といった方法です。ロジットプロセッサを無効にしたリクエストでは計算とメモリを節約するようにしてください

        * **注:** ラップしたリクエスト単位のロジットプロセッサでは、既定の `AdapterLogitsProcessor.update_state()` の実装により、`new_req_logits_processor()` がそのリクエストに対して `None` を返した場合にリクエスト単位のロジットプロセッサが無効になります

    3. **バッチレベルでロジットプロセッサを短絡（スキップ）する条件。** リクエスト単位でカスタムのロジットプロセッサを無効にする方法を定義したとしても、それを計算量の削減につなげるのは難しい場合があります。たとえば `update_state()` と `apply()` が永続バッチ全体を 1 コマンドで処理する効率的なベクトル化実装を使っている場合です。1 件のリクエストがロジットプロセッサを無効にしているというだけで、`apply()` のベクトル化演算全体をスキップすることはできません。実行中のどのリクエストもそのカスタムのロジットプロセッサを使っていないという端のケースで計算を節約するには、すべてのリクエストでロジットプロセッサが無効な場合に `apply()` が入力テンソルをそのまま返すよう設計することを推奨します。同様に、どのリクエストもロジットプロセッサを有効にしていない場合に `update_state()` の処理をスキップできないか検討してください

        * さらに、`update_state()` で計算を節約する簡単な方法は、`batch_update` が `None` のときに早期リターンすることです

        * **注:** ラップしたリクエスト単位のロジットプロセッサでは、`AdapterLogitsProcessor` の基底クラスが上記の最適化を既定で実装しています

* ロジットプロセッサの `update_state` メソッドが、完了したリクエスト（Add により置き換えられた、あるいは Remove の対象となったリクエスト）の情報を確実に破棄するようにしてください

    * **注:** ラップしたリクエスト単位のロジットプロセッサでは、`AdapterLogitsProcessor` の基底クラスがこれを既定で処理します

* ロジットプロセッサの挙動が一貫している場合、`is_argmax_invariant()` は `True` または `False` にハードコードできます。ただし、argmax 不変性はプログラム的に判定することもできます（たとえば、ロジットプロセッサがユーザーによってカスタマイズ可能で、それが argmax 不変性に影響する場合など）。このため、`is_argmax_invariant()` はクラスメソッドではありません
