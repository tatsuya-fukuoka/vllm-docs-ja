# プラグインシステム { #plugin-system }

コミュニティからは、独自機能で vLLM を拡張したいという要望が頻繁に寄せられます。これに応えるため、vLLM には vLLM のコードベースを変更せずに独自機能を追加できるプラグインシステムが用意されています。このドキュメントでは、vLLM におけるプラグインの仕組みと、vLLM 向けプラグインの作り方を説明します。

## vLLM におけるプラグインの仕組み { #how-plugins-work-in-vllm }

プラグインは、ユーザーが登録し vLLM が実行するコードです。vLLM のアーキテクチャ（[アーキテクチャ概要](arch_overview.md)を参照）を考えると、特にさまざまな並列化手法を用いた分散推論では複数のプロセスが関わります。プラグインを正しく有効にするには、vLLM が作成するすべてのプロセスがそのプラグインを読み込む必要があります。これは `vllm.plugins` モジュールの [`load_plugins_by_group`](https://docs.vllm.ai/en/v0.26.0/api/vllm/plugins/#vllm.plugins.load_plugins_by_group) 関数によって行われます。

## vLLM がプラグインを見つける仕組み { #how-vllm-discovers-plugins }

vLLM のプラグインシステムは、標準の Python の `entry_points` の仕組みを使います。この仕組みにより、開発者は自分の Python パッケージ内の関数を、他のパッケージから利用できるよう登録できます。プラグインの例を示します。

??? code

    ```python
    # inside `setup.py` file
    from setuptools import setup

    setup(name='vllm_add_dummy_model',
        version='0.1',
        packages=['vllm_add_dummy_model'],
        entry_points={
            'vllm.general_plugins':
            ["register_dummy_model = vllm_add_dummy_model:register"]
        })

    # inside `vllm_add_dummy_model/__init__.py` file
    def register():
        from vllm import ModelRegistry

        if "MyLlava" not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(
                "MyLlava",
                "vllm_add_dummy_model.my_llava:MyLlava",
            )
    ```

パッケージへのエントリポイントの追加については、[公式ドキュメント](https://setuptools.pypa.io/en/latest/userguide/entry_point.html)を参照してください。

すべてのプラグインは 3 つの要素からなります。

1. **プラグイングループ**: エントリポイントグループの名前です。vLLM は一般的なプラグインを登録するために、エントリポイントグループ `vllm.general_plugins` を使います。これは `setup.py` の `entry_points` のキーにあたります。vLLM の一般的なプラグインでは常に `vllm.general_plugins` を使ってください。
2. **プラグイン名**: プラグインの名前です。これは `entry_points` の辞書内の値にあたります。上の例では、プラグイン名は `register_dummy_model` です。プラグインは環境変数 `VLLM_PLUGINS` によって名前で絞り込めます。特定のプラグインだけを読み込むには、`VLLM_PLUGINS` にそのプラグイン名を設定します。
3. **プラグインの値**: プラグインシステムに登録する関数またはモジュールの完全修飾名です。上の例では、プラグインの値は `vllm_add_dummy_model:register` で、`vllm_add_dummy_model` モジュール内の `register` という関数を指します。

## サポートされるプラグインの種類 { #types-of-supported-plugins }

- **一般プラグイン**（グループ名 `vllm.general_plugins`）: 主な用途は、ツリー外の独自モデルを vLLM に登録することです。プラグイン関数の中で `ModelRegistry.register_model` を呼び出してモデルを登録します。公式のモデルプラグインの例としては、`BartForConditionalGeneration` のサポートを追加する [bart-plugin](https://github.com/vllm-project/bart-plugin) を参照してください。

- **プラットフォームプラグイン**（グループ名 `vllm.platform_plugins`）: 主な用途は、ツリー外の独自プラットフォームを vLLM に登録することです。プラグイン関数は、現在の環境でそのプラットフォームがサポートされていない場合は `None` を、サポートされている場合はプラットフォームクラスの完全修飾名を返します。

- **IO プロセッサプラグイン**（グループ名 `vllm.io_processor_plugins`）: 主な用途は、プーリングモデルに対するモデルプロンプトとモデル出力の独自の前処理・後処理を登録することです。プラグイン関数は IOProcessor のクラスの完全修飾名を返します。

- **統計ロガープラグイン**（グループ名 `vllm.stat_logger_plugins`）: 主な用途は、ツリー外の独自ロガーを vLLM に登録することです。エントリポイントは StatLoggerBase を継承したクラスである必要があります。

- **エンドポイントプラグイン**（グループ名 `vllm.endpoint_plugins`）: 主な用途は、OpenAI 互換 API サーバーにツリー外の独自 HTTP ルートを登録することです。上記の他のプラグイングループとは異なり、エンドポイントプラグインは API サーバーのフロントエンドプロセスでのみ読み込まれ、**既定では読み込まれません**。インターフェースについては[エンドポイントプラグイン](endpoint_plugins.md)を、オプトインと信頼モデルについては[セキュリティ](../usage/security.md#endpoint-plugins)を参照してください。

## プラグインを書く際の指針 { #guidelines-for-writing-plugins }

- **再入可能であること**: エントリポイントに指定する関数は再入可能、つまり複数回呼び出されても問題が起きないようにしてください。一部のプロセスでは関数が複数回呼ばれる可能性があるため、これが必要です。

### プラットフォームプラグインの指針 { #platform-plugins-guidelines }

1. プラットフォームプラグインのプロジェクト（例: `vllm_add_dummy_platform`）を作成します。プロジェクトの構成は次のようになります。

    ```shell
    vllm_add_dummy_platform/
    ├── vllm_add_dummy_platform/
    │   ├── __init__.py
    │   ├── my_dummy_platform.py
    │   ├── my_dummy_worker.py
    │   ├── my_dummy_attention.py
    │   ├── my_dummy_device_communicator.py
    │   ├── my_dummy_custom_ops.py
    ├── setup.py
    ```

2. `setup.py` に次のエントリポイントを追加します。

    ```python
    setup(
        name="vllm_add_dummy_platform",
        ...
        entry_points={
            "vllm.platform_plugins": [
                "my_dummy_platform = vllm_add_dummy_platform:register"
            ]
        },
        ...
    )
    ```

    `vllm_add_dummy_platform:register` が呼び出し可能な関数であり、プラットフォームクラスの完全修飾名を返すことを確認してください。例:

    ```python
    def register():
        return "vllm_add_dummy_platform.my_dummy_platform.MyDummyPlatform"
    ```

3. `my_dummy_platform.py` にプラットフォームクラス `MyDummyPlatform` を実装します。プラットフォームクラスは `vllm.platforms.interface.Platform` を継承する必要があります。インターフェースに従って関数を 1 つずつ実装してください。少なくとも実装すべき重要な関数とプロパティは次のとおりです。

    - `_enum`: [`PlatformEnum`](https://docs.vllm.ai/en/v0.26.0/api/vllm/platforms/interface/#vllm.platforms.interface.PlatformEnum) によるデバイスの列挙値を表すプロパティです。通常は、ツリー外のプラットフォームを意味する `PlatformEnum.OOT` にします。
    - `device_type`: PyTorch が使うデバイスの種類を返すプロパティです。たとえば `"cpu"`、`"cuda"` などです。
    - `device_name`: 通常は `device_type` と同じ値を設定します。主にログ出力の用途で使われます。
    - `check_and_update_config`: vLLM の初期化プロセスの非常に早い段階で呼ばれる関数です。プラグインが vLLM の設定を更新するために使います。たとえば、ブロックサイズやグラフモードの設定などをこの関数で更新できます。最も重要なのは、ワーカープロセスでどのワーカークラスを使うかを vLLM に伝えるため、この関数で **worker_cls** を設定することです。
    - `get_attn_backend_cls`: Attention バックエンドクラスの完全修飾名を返します。
    - `get_device_communicator_cls`: デバイスコミュニケータクラスの完全修飾名を返します。

4. `my_dummy_worker.py` にワーカークラス `MyDummyWorker` を実装します。ワーカークラスは [`WorkerBase`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/worker_base/#vllm.v1.worker.worker_base.WorkerBase) を継承する必要があります。インターフェースに従って関数を 1 つずつ実装してください。基本的に、基底クラスのすべてのインターフェースは vLLM のさまざまな箇所から呼ばれるため実装すべきです。モデルを実行できるようにするために最低限実装すべき関数は次のとおりです。

    - `init_device`: ワーカーのデバイスをセットアップするために呼ばれます。
    - `initialize_cache`: ワーカーのキャッシュ設定を行うために呼ばれます。
    - `load_model`: モデルの重みをデバイスに読み込むために呼ばれます。
    - `get_kv_cache_spec`: モデルの KV キャッシュ仕様を生成するために呼ばれます。
    - `determine_available_memory`: モデルのピークメモリ使用量をプロファイルし、OOM を起こさずに KV キャッシュへ割り当てられるメモリ量を判断するために呼ばれます。
    - `initialize_from_config`: 指定された kv_cache_config でデバイス側の KV キャッシュを確保するために呼ばれます。
    - `execute_model`: モデルの推論のために毎ステップ呼ばれます。

    追加で実装できる関数は次のとおりです。

    - スリープモード機能をサポートしたい場合は、`sleep` と `wakeup` を実装してください。
    - グラフモード機能をサポートしたい場合は、`compile_or_warm_up_model` を実装してください。
    - 投機的デコーディング機能をサポートしたい場合は、`take_draft_token_ids` を実装してください。
    - LoRA 機能をサポートしたい場合は、`add_lora`、`remove_lora`、`list_loras`、`pin_lora` を実装してください。
    - データ並列機能をサポートしたい場合は、`execute_dummy_batch` を実装してください。

    実装できる関数の詳細は、ワーカーの基底クラス [`WorkerBase`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/worker/worker_base/#vllm.v1.worker.worker_base.WorkerBase) を参照してください。

5. `my_dummy_attention.py` に Attention バックエンドクラス `MyDummyAttention` を実装します。Attention バックエンドクラスは [`AttentionBackend`](https://docs.vllm.ai/en/v0.26.0/api/vllm/v1/attention/backend/#vllm.v1.attention.backend.AttentionBackend) を継承する必要があります。これは自分のデバイスで Attention を計算するために使われます。`vllm.v1.attention.backends` には多くの Attention バックエンドの実装が含まれているので、参考にしてください。

6. 高い性能を得るためにカスタム op を実装します。ほとんどの op は PyTorch のネイティブ実装で動作しますが、性能が十分でない場合があります。その場合は、プラグイン向けに専用のカスタム op を実装できます。現在 vLLM がサポートするカスタム op の種類は次のとおりです。

    - PyTorch の op
      PyTorch の op には 3 種類あります。

        - `communicator ops`: デバイスコミュニケータの op です。all-reduce、all-gather などが該当します。
          `my_dummy_device_communicator.py` にデバイスコミュニケータクラス `MyDummyDeviceCommunicator` を実装してください。デバイスコミュニケータクラスは [`DeviceCommunicatorBase`](https://docs.vllm.ai/en/v0.26.0/api/vllm/distributed/device_communicators/base_device_communicator/#vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase) を継承する必要があります。
        - `common ops`: 一般的な op です。matmul、softmax などが該当します。
          ツリー外（oot）として登録する方法で実装してください。詳細は [`CustomOp`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/custom_op/#vllm.model_executor.custom_op.CustomOp) クラスを参照してください。
        - `csrc ops`: C++ の op です。C++ で実装され、torch のカスタム op として登録されます。
          csrc モジュールと `vllm._custom_ops` を参考に自分の op を実装してください。

    - Triton の op
      Triton の op については、現時点でカスタムの方法は使えません。

7. （任意）LoRA、グラフバックエンド、量子化、Mamba の Attention バックエンドなど、その他の差し替え可能なモジュールを実装します。

## 互換性の保証 { #compatibility-guarantee }

vLLM は、`ModelRegistry.register_model` のようにドキュメント化されたプラグインのインターフェースが、モデル登録のために常に利用可能であることを保証します。ただし、プラグインが対象とする vLLM のバージョンとの互換性を担保するのは、プラグイン開発者の責任です。たとえば `"vllm_add_dummy_model.my_llava:MyLlava"` は、そのプラグインが対象とする vLLM のバージョンと互換である必要があります。

モデルやモジュールのインターフェースは、vLLM の開発の過程で変わる可能性があります。非推奨のログが表示された場合は、プラグインを最新版に更新してください。

## 非推奨のお知らせ { #deprecation-announcement }

!!! warning "非推奨"
    - `Platform.get_attn_backend_cls` の `use_v1` パラメータは非推奨です。v0.13.0 で削除されました。
    - `vllm.attention` の `_Backend` は非推奨です。v0.13.0 で削除されました。新しい Attention バックエンドを `AttentionBackendEnum` に追加するには、代わりに `vllm.v1.attention.backends.registry.register_backend` を使ってください。
    - `seed_everything` のプラットフォームインターフェースは非推奨です。v0.16.0 で削除されました。代わりに `vllm.utils.torch_utils.set_random_seed` を使ってください。
    - `Platform.validate_request` の `prompt` は非推奨です。v0.18.0 で削除されました。
