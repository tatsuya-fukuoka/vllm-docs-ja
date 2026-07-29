# ロジットプロセッサ { #logits-processors }

!!! important
    ロジットプロセッサの設計変更の一部はまだ進行中で、API は近い将来変わる可能性があります。この部分の API は近いうちに安定させたいと考えています。

このドキュメントでは、vLLM エンジンがロジットプロセッサとどのようにやり取りするか、そして vLLM がロジットプロセッサの実装に対してサポートするプログラミングモデルを説明します。

## ロジットプロセッサの背景 { #logits-processors-background }

ロジットプロセッサは次トークンの確率分布を調整するもので、通常はモデルを望ましい振る舞いへ誘導することを目的とします。

vLLM では、ロジットプロセッサはバッチ単位で動作します。あるエンジンステップにおいて、ロジットプロセッサはモデルが出力した生のロジットの `(num_requests) x (vocab_size)` テンソルを受け取ります。そのロジットプロセッサを有効にしているすべてのリクエストについて、対応するロジットテンソルの行に変換を適用し、それ以外の行は変更しません。変換後のロジットテンソルは softmax に渡されます。

## vLLM エンジンにおけるロジットプロセッサ { #logits-processors-in-the-vllm-engine }

vLLM エンジンの永続バッチ（persistent batch）のデータ構造は、読み込まれたロジットプロセッサのリストを保持します。

バッチ全体を一度に処理するため、各ロジットプロセッサはバッチ内のリクエストに関するメタデータ（すなわち、リクエストごとのロジットプロセッサ固有の設定）を保持することがあります。したがって、ロジットプロセッサはステートフルです。

各エンジンステップで、vLLM エンジンは (1) 各ロジットプロセッサの内部状態を更新し、(2) モデル出力のロジットにロジットプロセッサを適用します。

### ロジットプロセッサの内部状態の更新 { #updating-logits-processor-internal-state }

各エンジンステップの開始時、永続バッチはスケジューラの出力に応じてリクエストの追加・破棄・並べ替えを行うことがあります。永続バッチが再構成されたあと、vLLM エンジンは各ロジットプロセッサの `update_state()` メソッドを呼び出します。これは、エンジンステップ開始時の新しい永続バッチの状態に合わせて、ロジットプロセッサの内部状態を並べ替えるために必要です。

以下の疑似コードは、vLLM の永続バッチが各ロジットプロセッサにバッチ状態の変更を通知する流れを示しています。

??? code "モデルランナーがロジットプロセッサの状態を更新する"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            self._update_states(scheduler_output)

            ...

        def _update_states(...):

            ...

            # ...update persistent batch to reflect new/finished requests & reordering
            # of requests within batch...

            ...

            self.input_batch.refresh_metadata()


    # gpu_input_batch.py

    class InputBatch:

        ...

        def refresh_metadata(self):

            ...

            # Update each logits processor's state to reflect persistent batch state
            batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
            for logit_proc in self.logitsprocs.all:
                logit_proc.update_state(batch_update)

            ...


    # vllm/v1/sample/logits_processor/interface.py

    @dataclass(frozen=True)
    class BatchUpdate:
        # Batch state-change data structure which is passed to logits processors'
        # update_state() methods

        batch_size: int

        removed: Sequence[RemovedRequest]
        added: Sequence[AddedRequest]
        moved: Sequence[MovedRequest]
    
    ```

### モデル出力のロジットへのロジットプロセッサの適用 { #applying-logits-processors-to-the-model-output-logits }

永続バッチの状態を更新したあと、vLLM のモデルランナーはモデル推論を実行してロジットを得ます。続いて、モデルランナーはそのロジットに対してサンプラーを呼び出します。サンプラーの処理の一部として、モデル出力のロジットに対してロジットプロセッサの `apply()` メソッドが呼ばれ、変換後のロジットが得られます（`apply()` はロジットをインプレースでもアウトオブプレースでも変更できますが、インプレースのほうがメモリ効率に優れます）。この流れを以下の疑似コードに示します。

サンプラーは `SamplingMetadata.logitsprocs` を通じてロジットプロセッサにアクセスします。vLLM エンジンが `SamplingMetadata` を構築する際（以下のコードには示していません）、ロジットプロセッサのリストへの参照が永続バッチのデータ構造から `SamplingMetadata` へ渡されます。

??? code "モデル出力のロジットにロジットプロセッサを適用する"

    ``` python
    # gpu_model_runner.py

    class GPUModelRunner(...):

        ...

        def execute_model(self, scheduler_output, ...):
            # (discussed in previous section)
            self._update_states(scheduler_output)

            ...

            # ...run model inference to obtain logits...

            ...

            # Invoke sampler, which applies logits processors
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=sampling_metadata)

            ...


    # sampler.py

    class Sampler(nn.Module):

        ...

        def forward(self, logits, sampling_metadata):

            ...

            # Apply non-argmax-invariant logits processors to model output logits
            for processor in (sampling_metadata.logitsprocs.non_argmax_invariant):
                logits = processor.apply(logits)

            sampled = self.sample(logits, sampling_metadata)

            ...

            # ...return sampler output data structure...


        def sample(self, logits, sampling_metadata)

            ...

            # ...exit early if all requests are greedy-sampling...

            ...

            # Apply argmax-invariant logits processors
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                logits = processor.apply(logits)

            ...

            # ...perform sampling and return sampling result...
    ``` 

サンプリング時、サンプラーは永続バッチ内のすべてのリクエストが貪欲サンプリングを使っているかを確認します。そうであれば、サンプラーは「argmax 不変」のロジットプロセッサをスキップして計算を節約します。ここで「argmax」とは、ロジットテンソルのある行で最も高いロジット値を持つトークン ID（つまり、そのリクエストについてモデルが最も高く評価したトークン）を指す略語です。

* **argmax 不変のロジットプロセッサ**とは、argmax を変えないロジットプロセッサ（Min-P など）です。たとえば、確率の最も低いトークンをマスクするロジットプロセッサは、どのトークン ID が最大のロジットを持つかを変えません。貪欲サンプリングは常に最大のロジット値を持つトークン ID を選ぶため、概念上、argmax 不変のロジットプロセッサは貪欲サンプリングのリクエストではスキップできます。

* **argmax 不変でないロジットプロセッサ**とは、argmax を変える可能性があるロジットプロセッサです。たとえば、デコードを強制的に終了させるために一定ステップ数を過ぎたら EOS 以外のすべてのトークンをマスクするロジットプロセッサは、最大ロジット値のトークンをマスクしてしまい、結果として argmax を変えることがあります。概念上、こうしたロジットプロセッサは貪欲サンプリングのリクエストでもスキップできません。

vLLM のロジットプロセッサの抽象は、エンジンがバッチ単位でロジットプロセッサを適用することを前提としています。したがって実際には、argmax 不変のロジットプロセッサをスキップできるのは、バッチ全体が貪欲サンプリングを使っている場合だけです。

## ロジットプロセッサのプログラミングモデル { #logits-processor-programming-model }

前の節では、vLLM のロジットプロセッサがサポートすべきインターフェースに触れました。この節では、vLLM エンジンと互換なロジットプロセッサを実装するためのプログラミングモデルを、`LogitsProcessor` 基底クラスとそのインターフェースメソッド、および永続バッチの状態変化を表す `BatchUpdate` データ構造を含めて全体的に説明します。いずれも以下のコードに示します。

??? code "`LogitsProcessor` 基底クラスと `BatchUpdate` データ構造"

    ``` python
    from abc import ABC, abstractmethod
    from collections.abc import Sequence
    from dataclasses import dataclass
    from enum import Enum, auto
    from typing import TYPE_CHECKING

    import torch

    from vllm import SamplingParams

    if TYPE_CHECKING:
        from vllm.config import VllmConfig


    class MoveDirectionality(Enum):
        # One-way i1->i2 req move within batch
        UNIDIRECTIONAL = auto()
        # Two-way i1<->i2 req swap within batch
        SWAP = auto()


    # (index, params, prompt_tok_ids, output_tok_ids) tuples for new
    # requests added to the batch.
    AddedRequest = tuple[int, SamplingParams, list[int], list[int]]

    # (index 1, index 2, directionality) tuples representing
    # one-way moves or two-way swaps of requests in batch
    MovedRequest = tuple[int, int, MoveDirectionality]

    # Batch indices of any removed requests.
    RemovedRequest = int


    @dataclass(frozen=True)
    class BatchUpdate:
        """Persistent batch state change info for logitsprocs"""
        batch_size: int  # Current num reqs in batch

        # Metadata for requests added to, removed from, and moved
        # within the persistent batch.
        #
        # Key assumption: the `output_tok_ids` list (which is an element of each
        # tuple in `added`) is a reference to the request's running output tokens
        # list; via this reference, the logits processors always see the latest
        # list of generated output tokens
        removed: Sequence[RemovedRequest]
        moved: Sequence[MovedRequest]
        added: Sequence[AddedRequest]


    class LogitsProcessor(ABC):

        @abstractmethod
        def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                    is_pin_memory: bool) -> None:
            raise NotImplementedError

        @abstractmethod
        def apply(self, logits: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @abstractmethod
        def is_argmax_invariant(self) -> bool:
            """True if logits processor has no impact on the
            argmax computation in greedy sampling.
            NOTE: may or may not have the same value for all
            instances of a given LogitsProcessor subclass,
            depending on subclass implementation.
            """
            raise NotImplementedError

        @abstractmethod
        def update_state(
            self,
            batch_update: "BatchUpdate" | None,
        ) -> None:
            """Called when there are new output tokens, prior
            to each forward pass.

            Args:
                batch_update is non-None iff there have been
                changes to the batch makeup.
            """
            raise NotImplementedError

        @classmethod
        def validate_params(cls, sampling_params: SamplingParams):
            """Validate sampling params for this logits processor.

            Raise ValueError for invalid ones.
            """
            return None

    ```

vLLM のロジットプロセッサは `LogitsProcessor` を継承し、少なくとも次のメソッドを定義する必要があります。

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

* `update_state(self, batch_update: "BatchUpdate" | None) -> None`:
    * 現在のエンジンステップ開始時の永続バッチの状態変化を表す `BatchUpdate` データ構造を受け取ります
    * `BatchUpdate` のメンバを使ってロジットプロセッサの内部状態を更新します
    * **注:** バッチ更新のデータ構造は `None` になることがあり、これはバッチの構成に変化がないことを示します。この場合でも、ロジットプロセッサは、追加時に保持した `output_token_ids` のリストが更新されているのを踏まえて状態を更新したい場合があります。

* `validate_params(cls, sampling_params: SamplingParams)`:
    * ロジットプロセッサが使う `SamplingParams` の引数（とくにカスタム引数）が不正な場合に `ValueError` を送出します。
    * リクエストがエントリポイントに送られると、`validate_params()` が `SamplingParams` を検証し、不正な引数を含むリクエストを拒否します。

### `BatchUpdate` データ構造 { #batchupdate-data-structure }

`BatchUpdate` の抽象は、永続バッチをリクエストのリストとしてモデル化し、バッチ状態を変更する次の操作をサポートします（以下で操作を挙げる順序は、`update_state()` 内でそれらを処理すべき順序を反映しています）。

* **Remove:** インデックス `i` のリクエストを（置き換えなしで）削除します

    * Remove は `Batchupdate.removed` の `int`（`i` を表す）として表現されます

    * インデックス指定の削除がバッチに与える影響:

        ``` text
        バッチ: [A,B,C]
        Remove @ i:  1

        =>

        新しいバッチ: [A,x,C] # B を破棄し、空きスロットを残す
        ```

* **Add:** インデックス `i` に新しいリクエストを追加（または既存のリクエストを置き換え）します。リクエストが置き換えられた場合、それに紐づく状態は破棄すべきです。

    * Add は `Batchupdate.added` のタプルとして表現されます

        ``` text
        (インデックス, 新しいリクエストの SamplingParams, プロンプトのトークン ID, 出力のトークン ID)
        ```

    * `プロンプトのトークン ID` と `出力のトークン ID` は、それぞれリクエストのプロンプトトークン ID リストと出力トークン ID リストへの参照です。出力トークン ID のリストはエンジンステップごとに伸びていき、参照渡しであるためロジットプロセッサからその伸長が見えます。**これは、これまでに生成されたトークンを考慮するロジットプロセッサにとって重要です**。

    * 追加されたリクエストのタプルのフィールドを内部表現へどう取り込むか（あるいは取り込まないか）は、個々のロジットプロセッサのサブクラスの実装によって決まります。たとえば、プロンプトや出力のトークン ID を利用しないロジットプロセッサは、`index` と `SamplingParams` だけを使い、残りのフィールドを破棄すればよいでしょう

    * インデックス `i` に現在リクエストが入っている場合、置き換えが発生します。

        ``` text
        バッチ: [A,B,C]
        追加する新しいリクエスト @ i: D @ 1

        =>

        新しいバッチ: [A,D,C] # D を追加し、B を破棄
        ```

    * インデックス `i` に現在リクエストが入っていない場合（`i` が現在のバッチサイズの範囲外の場合）:

        ``` text
        バッチ: [A,B,C]
        追加する新しいリクエスト @ i: D @ 3

        =>

        新しいバッチ: [A,B,C,D] # D を追加し、バッチを拡張
        ```

* **Move:** インデックス `s` のリクエストをインデックス `d` へ移動する、またはインデックス `s` と `d` のリクエストを交換します

    * Move は `Batchupdate.moved` のタプルとして表現されます

        ``` text
        (s, d, UNIDIRECTIONAL または SWAP)
        ```

    * Move が `UNIDIRECTIONAL` を指定する場合:

        * インデックス `s` のリクエストがインデックス `d` へ移動し、インデックス `s` は空きスロットになります

            ``` text
            バッチ: [A,x,C,D]
            一方向の Move s -> d:  3 -> 1

            =>

            新しいバッチ: [A,D,C,x] # D を 1 へ移動し、3 に空きスロットを残す
            ```

        * インデックス `d` にすでに別のリクエストがあった場合、それは置き換えられて破棄されます

            ``` text
            バッチ: [A,B,C,D]
            一方向の Move s -> d:  3 -> 1

            =>

            新しいバッチ: [A,D,C,x] # D を 1 へ移動し、B を破棄して 3 に空きスロットを残す
            ```

    * Move が `SWAP` を指定する場合、`s` と `d` のリクエストがインデックスを交換します

        ``` text
        バッチ: [A,B,C,D]
        Swap の Move s <-> d:  3 <-> 1

        =>

        新しいバッチ: [A,D,C,B] # B と D を交換
        ```

さらに `BatchUpdate` データ構造には、エンジンステップ開始時点の永続バッチのサイズを表す `batch_size` が含まれます。

### vLLM エンジンが `BatchUpdate` データ構造を構築する方法 { #how-the-vllm-engine-builds-the-batchupdate-data-structure }

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

#### 例: 新規リクエストが完了リクエストより少ない場合のバッチ更新 { #example-batch-update-with-fewer-new-requests-than-finished-requests }

次の例は、新規リクエストが 1 件投入され、完了したリクエストが 2 件除かれ、さらに attention バックエンドがバッチの並び順を最適化するために交換を行うエンジンステップをモデル化したものです。

``` text
バッチの状態（エンジンステップ開始時）: [A,B,C,D]
バッチサイズ: 4

新規リクエスト: E

完了したリクエスト: A, C

処理の手順（BatchUpdate の抽象を使用）:

1. インデックス 0 に E を Add

[E,B,C,D] # A を破棄
バッチサイズ: 4

2. インデックス 2 を Remove

[E,B,x,D] # C を破棄、インデックス 2 が空きスロット
バッチサイズ: 4

3. 一方向の Move 3 -> 2 でバッチを圧縮し、バッチを縮小

[E,B,D] x # 空きスロットはバッチの外側になった
バッチサイズ: 3

4. attention バックエンドの最適化: Swap 0 <-> 1 でバッチを並べ替え

[B,E,D]
バッチサイズ: 3

```

得られる `BatchUpdate` データ構造は次のようになります。

``` text
BatchUpdate のインスタンス
* added: [(0, E の SamplingParams, E のプロンプトトークンへの参照, E の出力トークンへの参照)]
* removed: [2] # リクエスト C は置き換えなしで削除された
* moved: [(3,2,UNIDIRECTIONAL),(0,1,SWAP)]
```

#### 例: 新規リクエストが完了リクエストより多い場合のバッチ更新 { #example-batch-update-with-more-new-requests-than-finished-requests }

次の例は、新規リクエストが 2 件投入され、完了したリクエストが 1 件除かれ、さらに attention バックエンドがバッチの並び順を最適化するために交換を行うエンジンステップをモデル化したものです。

``` text
バッチの状態（エンジンステップ開始時）: [A,B,C,D]
バッチサイズ: 4

新規リクエスト: E,F

完了したリクエスト: C

処理の手順（BatchUpdate の抽象を使用）:

1. インデックス 2 に E を Add

[A,B,E,D] # C を破棄
バッチサイズ: 4

2. インデックス 4（現在の最大バッチインデックス + 1）に F を Add

[A,B,E,D,F] # バッチを 1 つ拡張
バッチサイズ: 5

4. attention バックエンドの最適化: Swap 0 <-> 1 でバッチを並べ替え

[B,A,E,D,F]
バッチサイズ: 5

```

Remove 操作による空きスロットが残らないため、バッチの圧縮はスキップされる点に注意してください。

得られる `BatchUpdate` データ構造は次のようになります。

``` text
BatchUpdate のインスタンス
* added: [(2, E の SamplingParams, E のプロンプトトークンへの参照, E の出力トークンへの参照),(4, F の SamplingParams, F のプロンプトトークンへの参照, F の出力トークンへの参照)]
* removed: [] # 置き換えなしで削除されたリクエストはない
* moved: [(0,1,SWAP)]
```

## vLLM に新しいロジットプロセッサを追加する方法 { #how-to-introduce-a-new-logits-processor-to-vllm }

### 組み込みロジットプロセッサを書く際のベストプラクティス { #best-practices-for-writing-built-in-logits-processors }

* ロジットプロセッサがバッチ単位で動作することを踏まえ、効率的な `apply()` と `update_state()` の実装を書いてください
    * たとえば、`apply()` の実装や `update_state()` での内部状態ベクトルの更新に、効率的なベクトル化演算を使えるかもしれません
    * ただし、そのロジットプロセッサが使われる頻度が低いと考えられる場合は、リクエスト状態を「疎な」表現で持つほうが適切なこともあります。すなわち、そのロジットプロセッサを有効にしているリクエストのメタデータだけを辞書で保持する、といった方法です

* 次の点はロジットプロセッサの作者が決めることです。

    1. **そのリクエストに対するロジットプロセッサの挙動を設定する、リクエストごとの属性。** たとえば vLLM 向けに新しい組み込みロジットプロセッサを書く場合、`SamplingParams` と vLLM の REST API にフィールドを追加する必要があるかもしれませんし、ないかもしれません

    2. **リクエストごとにロジットプロセッサを有効 / 無効にする条件。** 組み込みのロジットプロセッサを常にすべてのリクエストに作用させるつもりでない限り、あるリクエストについてロジットプロセッサを無効にできるように書くべきです。たとえば引数の既定値を `None` にする、あるいは何もしないことを表す特定の値（`0.0` など）を渡す、といった方法です。ロジットプロセッサを無効にしたリクエストでは計算とメモリを節約するようにしてください

    3. **バッチレベルでロジットプロセッサを短絡（スキップ）する条件。** リクエスト単位で組み込みロジットプロセッサを無効にする方法を定義したとしても、それを計算量の削減につなげるのは難しい場合があります。たとえば `update_state()` と `apply()` が永続バッチ全体を 1 コマンドで処理する効率的なベクトル化実装を使っている場合です。1 件のリクエストがロジットプロセッサを無効にしているというだけで、`apply()` のベクトル化演算全体をスキップすることはできません。実行中のどのリクエストもその組み込みロジットプロセッサを使っていないという端のケースで計算を節約するには、すべてのリクエストでロジットプロセッサが無効な場合に `apply()` が入力テンソルをそのまま返すよう設計することを推奨します。同様に、どのリクエストもロジットプロセッサを有効にしていない場合に `update_state()` の処理をスキップできないか検討してください

        * さらに、`update_state()` で計算を節約する簡単な方法は、batch_update が `None` のときに早期リターンすることです

* ロジットプロセッサの `update_state` メソッドが、完了したリクエスト（Add により置き換えられた、あるいは Remove の対象となったリクエスト）の情報を確実に破棄するようにしてください

* ロジットプロセッサの挙動が一貫している場合、`is_argmax_invariant()` は `True` または `False` にハードコードできます。ただし、argmax 不変性はプログラム的に判定することもできます（たとえば、ロジットプロセッサがユーザーによってカスタマイズ可能で、それが argmax 不変性に影響する場合など）。このため、`is_argmax_invariant()` はクラスメソッドではありません

### 組み込みのロジットプロセッサ { #built-in-logits-processors }

組み込みのロジットプロセッサは、vLLM エンジンの起動時に常に読み込まれます。新しい組み込みロジットプロセッサの書き方の例は、`vllm/v1/sample/logits_processor/builtin.py` にある既存の vLLM 組み込みロジットプロセッサを参照してください。幅広い利用者にとって有用と考えられる場合は、新しいロジットプロセッサを組み込みとして追加する PR を書く意義があります。vLLM は現在、上記のプログラミングモデルにもとづく次の組み込みロジットプロセッサを採用しています。

* Min-P

* ロジットバイアス

* Min-tokens

組み込みロジットプロセッサを書く際の参考として、これらの実装を確認してください。

さらに、次のロジットプロセッサ相当の機能はサンプラーにハードコードされており、まだ上記のプログラミングモデルを使っていません。これらの多くは、前述のロジットプロセッサのプログラミングモデルを使うようリファクタリングされる予定です。

* 許可するトークン ID

* Bad words

* 繰り返しペナルティ

* 頻度ペナルティ

* 出現ペナルティ

* Temperature

* Top-K

* Top-P

### カスタムのロジットプロセッサ { #custom-logits-processors }

vLLM は[ユーザー提供のカスタムロジットプロセッサ](../features/custom_logitsprocs.md)で拡張できます。
