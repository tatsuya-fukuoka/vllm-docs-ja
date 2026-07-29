# エンドポイントプラグイン { #endpoint-plugins }

エンドポイントプラグインを使うと、`vllm/entrypoints/openai/api_server.py` を編集することなく、ツリー外のパッケージから OpenAI 互換 API サーバーに HTTP ルートを追加できます。対象となるのは **HTTP の表面のみ**で、ルートの登録と、それらのルートが使うアプリケーション単位の状態（任意）です。プラグインは、ツリー内のサービングハンドラと同じ方法、つまり起動時に渡される `EngineClient` を通じてエンジンにアクセスします（例: `engine_client.collective_rpc(...)`）。新たなエンジンアクセス経路が導入されるわけではありません。

!!! warning "セキュリティ"
    エンドポイントプラグインは**既定では読み込まれず**、明示的に許可リストへ登録する必要があります。
    有効にする前に[エンドポイントプラグインのセキュリティ上の考え方](../usage/security.md#endpoint-plugins)を、
    特にルートの上書き（shadowing）に関する警告を必ず読んでください。

## `EndpointPlugin` プロトコル { #the-endpointplugin-protocol }

エンドポイントプラグインは、実行時にチェック可能な `Protocol` である [`EndpointPlugin`](https://docs.vllm.ai/en/v0.26.0/api/vllm/plugins/endpoint_plugins/interface/#vllm.plugins.endpoint_plugins.interface.EndpointPlugin) を実装します。

```python
class EndpointPlugin(Protocol):
    name: str
    required_tasks: tuple[SupportedTask, ...] | None

    def attach_router(self, app: FastAPI) -> None: ...

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None: ...
```

- `name`: ログや `VLLM_PLUGINS` の許可リストで使われる一意な識別子
- `required_tasks`: このプラグインを読み込むためにサーバーがサポートしていなければならないタスク。`None` はタスク要件がないことを意味します
- `attach_router`: `app` にルートを登録します
- `init_state`: ルートがリクエスト時に読み取る、アプリケーション単位の状態を初期化します

## 2 段階のライフサイクル { #the-two-phase-lifecycle }

ルートはエンジンが存在するより前に登録されます。そのため、このインターフェースはサーバー起動の異なる 2 つのタイミングで実行される 2 つのフックを公開しています。

| フェーズ | 呼び出し元 | `engine_client` は利用可能か | 行うこと |
| --- | --- | --- | --- |
| A. ルートの登録 | `build_app()` | 不可 | `attach_router(app)` でルートを追加します。ここでエンジンに触れてはいけません。 |
| B. 状態の初期化 | `init_app_state()` | 通常は可能。ただし CPU のみの render サーバーでは `None` | `init_state(engine_client, state, args)` で `engine_client` を保持するサービングハンドラを構築し、`state` に格納します。 |

`app.state` は `init_app_state()` に渡される `state` オブジェクト*そのもの*であるため、フェーズ A で格納したオブジェクトはフェーズ B から見え、フェーズ B で格納したオブジェクトはリクエスト時に `request.app.state` 経由でルートハンドラから見えます。これは、ツリー内のエンドポイントがすでに使っているのと同じパターンです。

### エンジンを持たないサーバー（render サーバー） { #engine-less-servers-the-render-server }

CPU のみの render サーバー（`init_render_app_state()`）には `EngineClient` がありません。それでも、`render` タスクの対象となるプラグイン（`required_tasks` が `None`、または `"render"` を含む）については両方のフェーズを実行します。`attach_router` は通常どおり呼ばれますが、`init_state` は `engine_client=None` で呼ばれます。

動作にエンジンを必要とするプラグインには 2 つの選択肢があります。

- `required_tasks` から `"render"` を除外し、そもそも render サーバーで読み込まれないようにする
- `render` で読み込まれることを受け入れたうえで、`init_state` またはルートハンドラで `None` を確認し、存在しないクライアントを参照する代わりにエラー応答（HTTP 503 など）を返す

`tests/plugins/vllm_add_dummy_endpoint_plugin` は 2 つ目の方法を示しています。そのルートハンドラは、`state.dummy_engine_client` が `None` のとき 503 を返します。

### ルートハンドラからエンジンにアクセスする { #reaching-the-engine-from-a-route-handler }

`init_state` は、プラグインが `engine_client` を小さなサービングハンドラに取り込み、`state` に格納する場所です。`attach_router` で追加したルートは、リクエスト時に `request.app.state` からそのハンドラを取り出し、それを通じて（多くの場合 `engine_client.collective_rpc(...)` で）エンジンを呼び出します。

次の最小限の例では、`required_tasks` が `None` であるため、簡潔さを優先して前節の `None` チェックを省略しています。実際にはこのプラグインは `render` の対象になるため、公開する前に `tests/plugins/vllm_add_dummy_endpoint_plugin` と同様に `engine_client=None` を扱うべきです。

```python
from fastapi import FastAPI, Request


class MyAdminEndpointPlugin:
    name = "my_admin_endpoint_plugin"
    required_tasks: tuple[str, ...] | None = None

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/plugins/my_admin_endpoint_plugin/scheduler_config")
        async def scheduler_config(raw_request: Request):
            engine_client = raw_request.app.state.my_engine_client
            results = await engine_client.collective_rpc("get_scheduler_config")
            return {"scheduler_config": results}

    async def init_state(self, engine_client, state, args) -> None:
        state.my_engine_client = engine_client
```

この例の完全かつテスト済みのバージョンは、リポジトリ内の `tests/plugins/vllm_add_dummy_endpoint_plugin` にあり、`tests/plugins_tests/test_endpoint_plugins.py` で（実際の HTTP リクエストを含む）エンドツーエンドのテストが行われています。

## エントリポイントの登録 { #registering-the-entry-point }

引数を取らないファクトリ（クラスまたは関数）を `vllm.endpoint_plugins` グループに登録します。ファクトリは `EndpointPlugin` を満たすオブジェクトを返す必要があります。

```toml
# pyproject.toml
[project.entry-points."vllm.endpoint_plugins"]
my_admin_api = "my_pkg.endpoints:MyAdminEndpointPlugin"
```

```python
# setup.py equivalent
setup(
    name="my_pkg",
    entry_points={
        "vllm.endpoint_plugins": [
            "my_admin_api = my_pkg.endpoints:MyAdminEndpointPlugin"
        ]
    },
)
```

エントリポイント名（上記の `my_admin_api`）は、プラグインの `name` 属性とは独立しています。`VLLM_PLUGINS` の許可リストは、`vllm.general_plugins` と同じ慣習に従い、**エントリポイント名**で照合されます（[プラグインシステム](plugin_system.md)を参照）。

## 制御: `VLLM_PLUGINS` と `required_tasks` { #gating-vllm_plugins-and-required_tasks }

エンドポイントプラグインは [`load_endpoint_plugins`](https://docs.vllm.ai/en/v0.26.0/api/vllm/plugins/#vllm.plugins.load_endpoint_plugins) によって検出・制御されます。これは他のプラグイングループで使われるローダーより厳格です。

- **`VLLM_PLUGINS` が設定され、そのプラグイン名が含まれていない限り、何も読み込まれません。** 他のプラグイングループでは、`VLLM_PLUGINS` で絞り込まない限りすべてが読み込まれます。エンドポイントプラグインはネットワークに露出する表面を追加するため、この既定を反転させています。[セキュリティ](../usage/security.md#endpoint-plugins)を参照してください。
- **`required_tasks` はサーバーがサポートするタスクと交差している必要があります**（`None` の場合を除く）。これにより、そのルートを提供できないサーバー（プーリング専用のデプロイなど）にルートが登録されるのを防げます。
- インスタンス化の途中で問題を送出したファクトリは、ログに記録されたうえでスキップされます。サーバーの起動が中断されることはありません。

エンドポイントプラグインを読み込むのはフロントエンドの API サーバープロセスだけです。ワーカーやエンジンコアのプロセスを気にする必要はありません。

## `vllm.general_plugins` との組み合わせ { #pairing-with-vllmgeneral_plugins }

エンドポイントプラグインが担うのは HTTP の表面のみです。エンジン側の新しい挙動（ワーカー側の新しい RPC メソッド、独自の統計値など）も必要な場合、その部分は既存の `vllm.general_plugins` グループを通じて別途提供します。こちらはワーカープロセスで読み込まれます（[プラグインシステム](plugin_system.md)を参照）。2 つのエントリポイントは**独立して**登録・読み込みされ、一方が他方を含意することはありません。推奨される配布形態は、両方を公開する 1 つのパッケージです。

```toml
[project.entry-points."vllm.general_plugins"]
my_admin_engine = "my_pkg.engine:register"      # adds the worker side method

[project.entry-points."vllm.endpoint_plugins"]
my_admin_api = "my_pkg.endpoints:MyAdminEndpointPlugin"  # adds the HTTP route
```

1 つのエンドポイントプラグインがエンジンやワーカーの状態も変更できると期待しないでください。ルートがまだ存在しないワーカー側のメソッドを必要とする場合は、対になる `general_plugins` のエントリポイントで追加してください。

## パス接頭辞の慣習 { #path-prefix-convention }

現時点では、ルートの衝突を防ぐ仕組みはありません（RFC [#46565](https://github.com/vllm-project/vllm/issues/46565) のフォローアップとして追跡されています）。プラグインの `attach_router` はコアのルートと衝突するパスを登録でき、あとから登録されたルートが優先されます。運用者を驚かせないために次のようにしてください。

- `/v1/...` などのコアの接頭辞を再利用せず、`/plugins/<plugin-name>/...` のような独自の接頭辞の下にルートを配置してください
- コアの接頭辞の下にルートを登録するのは（例で示した `/v1/admin/scheduler_config` のように）既存の挙動を上書きまたは拡張する明確な意図がある場合のみにし、プラグインを許可リストに追加する運用者向けにその旨を明記してください

## 互換性 { #compatibility }

`state` やサービングハンドラの内部（ツリー内の `OpenAIServing*` クラスの形など）は、まだ安定した公開契約ではありません。自己責任で利用するものとして扱い、vLLM のバージョン間で変わりうると想定してください。サポートされる表面は `FastAPI`、`EngineClient`、および `EndpointPlugin` プロトコルそのものです。
