# vLLM への貢献 { #contributing-to-vllm }

vLLM への貢献に関心を持っていただきありがとうございます。私たちのコミュニティはすべての人に開かれており、大小を問わずあらゆる貢献を歓迎します。プロジェクトに貢献する方法はいくつかあります。

- 問題やバグを見つけて報告する。
- 新しいモデルのサポートを要望する、または追加する。
- 新機能を提案する、または実装する。
- ドキュメントを改善する、あるいは How-to ガイドを寄稿する。

私たちはコミュニティによる支え合いの力も信じています。質問への回答、PR のレビュー、他の人への手助けも、非常に価値ある貢献として高く評価されます。

最後に、最も影響の大きい支援の 1 つは vLLM の認知を広めることです。ブログ記事で vLLM について書き、それがあなたの素晴らしいプロジェクトをどう支えているかを紹介してください。vLLM を使っているなら SNS で応援を表明したり、単にリポジトリにスターを付けて感謝を示したりするだけでも構いません。

## ジョブボード { #job-board }

どこから始めればよいか分からない場合は、次のリンクから取り組めるタスクを探してみてください。

- [Good first issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
    - [厳選されたオンボーディング用タスク](https://github.com/orgs/vllm-project/projects/6)
- [新しいモデルの要望](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22new-model%22)
    - [マルチモーダル対応のモデル](https://github.com/orgs/vllm-project/projects/10)

## ライセンス { #license }

[LICENSE](../../LICENSE) を参照してください。

## 開発 { #developing }

vLLM への貢献の第一歩は、GitHub リポジトリをクローンすることです。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

次に、Python の仮想環境を設定します。

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

vLLM の Python コードだけを開発する場合は、次のコマンドで vLLM をインストールします。

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

Rust フロントエンドのバイナリだけを再ビルドするには次のようにします。

```bash
./build_rust.sh          # release build
./build_rust.sh --debug  # faster build for development
```

vLLM の Python と CUDA/C++ の両方のコードを開発する場合は、まず PyTorch をインストールします。

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
```

次に、`requirements/build/cuda.txt` から必要なビルド依存パッケージをインストールします。`torch` は前の手順でインストール済みなので除外します。

```bash
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
```

最後に、次のコマンドで vLLM をインストールします。

```bash
uv pip install -e . --no-build-isolation
```

ソースからのインストールや他のハードウェア向けのインストールの詳細は、お使いのハードウェアの[インストール手順](../getting_started/installation/README.md)を確認し、「ソースから wheel をビルドする」のセクションを参照してください。

C++/CUDA カーネルを繰り返し開発する際に最適化されたワークフローについては、[インクリメンタルビルドのワークフロー](./incremental_build.md)の推奨事項を参照してください。

!!! tip
    vLLM は Python 3.10 から 3.13 に対応しています。ただし、vLLM の既定の [Dockerfile](../../docker/Dockerfile) には Python 3.12 が同梱され、CI のテスト（`mypy` を除く）も Python 3.12 で実行されます。

    そのため、ローカル環境と CI 環境の食い違いを最小限にするには、Python 3.12 での開発を推奨します。

### Lint { #linting }

vLLM はコードベースの lint とフォーマットに `pre-commit` を使っています。`pre-commit` が初めての場合は <https://pre-commit.com/#usage> を参照してください。`pre-commit` のセットアップは次のように簡単です。

```bash
uv pip install pre-commit>=4.5.1
pre-commit install
```

これで、コミットのたびに vLLM の `pre-commit` フックが自動的に実行されます。

!!! tip "ヒント"
    `pre-commit` フックは次のコマンドで手動実行できます。

    ```bash
    pre-commit run     # runs on staged files
    pre-commit run -a  # runs on all files (short for --all-files)
    ```

    ---

    一部の `pre-commit` フックは CI でのみ実行されます。必要であれば、次のようにローカルで実行できます。

    ```bash
    pre-commit run --hook-stage manual mypy-3.11
    ```

### ドキュメント { #documentation }

MkDocs は、プロジェクトのドキュメント作成に向いた、高速でシンプル、そして見た目も美しい静的サイトジェネレータです。ドキュメントのソースファイルは Markdown で書かれ、1 つの YAML 設定ファイル [mkdocs.yaml](../../mkdocs.yaml) で設定します。

まずは次のコマンドから始めます。

```bash
uv pip install -r requirements/docs.txt
```

!!! tip
    Python のバージョンがプラグインと互換であることを確認してください
    （たとえば `mkdocs-awesome-nav` は Python 3.10 以上が必要です）。

MkDocs には組み込みの開発サーバーがあり、作業しながらドキュメントをプレビューできます。リポジトリのルートで次を実行してください。

```bash
mkdocs serve                           # with API ref (~10 minutes)
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # API ref off (~15 seconds)
```

ログに `Serving on http://127.0.0.1:8000/` と表示されたら、ライブプレビューの準備完了です。ブラウザで <http://127.0.0.1:8000/> を開いて確認してください。

その他の機能や高度な設定については、次を参照してください。

- [MkDocs のドキュメント](https://www.mkdocs.org/)
- [Material for MkDocs のドキュメント](https://squidfunk.github.io/mkdocs-material/)（vLLM が使っている MkDocs のテーマ）

### テスト { #testing }

vLLM はコードベースのテストに `pytest` を使っています。

```bash
# Install the test dependencies used in CI (CUDA only)
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# Install some common test dependencies (hardware agnostic)
uv pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run tests for a single test file with detailed output
pytest -s -v tests/test_logger.py
```

!!! tip "Python.h が見つからない場合は python3-dev をインストール"
    上記のいずれかのコマンドが `Python.h: No such file or directory` で失敗する場合は、
    `sudo apt install python3-dev` で `python3-dev` をインストールしてください。

!!! warning "注意"
    現時点では、リポジトリ全体が `mypy` でチェックされているわけではありません。

    ---

    現時点では、CPU プラットフォームで実行した場合にすべてのユニットテストが通るわけではありません。
    ローカルでユニットテストを実行できる GPU 環境がない場合は、当面は継続的インテグレーションの
    システムにテストの実行を任せてください。

## Issue { #issues }

バグに遭遇した場合や機能の要望がある場合は、まず[既存の issue を検索](https://github.com/vllm-project/vllm/issues?q=is%3Aissue)して、すでに報告されていないか確認してください。報告されていない場合は、できるだけ多くの関連情報を添えて[新しい issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)してください。

!!! important
    セキュリティ脆弱性を発見した場合は、[こちら](../../SECURITY.md)の手順に従ってください。

## Pull Request とコードレビュー { #pull-requests-code-reviews }

vLLM への貢献ありがとうございます。PR を提出する前に、次の基準を満たしていることを確認してください。これは vLLM のコード品質を保ち、レビュープロセスの効率を高めるのに役立ちます。

### DCO と Signed-off-by { #dco-and-signed-off-by }

このプロジェクトに変更を貢献する際は、[DCO](../../DCO) に同意する必要があります。コミットには、DCO の条項への同意を証明する `Signed-off-by:` ヘッダーを含めなければなりません。

`git commit` に `-s` を付けると、このヘッダーが自動的に追加されます。

!!! tip
    IDE から自動的に sign-off を有効にすることもできます。

    - **PyCharm**: `Commit` ウィンドウで `Commit and Push...` ボタンの右にある `Show Commit Options` アイコンをクリックします。
      `git` のウィンドウが開き、`Author` を変更したり `Sign-off commit` を有効にしたりできます。
    - **VSCode**: [設定エディタ](https://code.visualstudio.com/docs/configure/settings)を開き、
      `Git: Always Sign Off`（`git.alwaysSignOff`）を有効にします。

### AI を活用した貢献 { #ai-assisted-contributions }

AI を活用した貢献を行う前に、次のことを守ってください。

1. **自分が関与する**: 「純粋なエージェント任せ」の PR を提出しないでください。変更されたすべての行をレビューし、エンドツーエンドで挙動を検証し、関連するテストを実行するのは人間の提出者の責任です。
2. **意義を確認する**: 単発の「作業のための作業」的な PR（タイポ 1 件、局所的なスタイル修正、可変デフォルト引数 1 件の修正など）は避けてください。機械的な整理は、明確で体系的な範囲にまとめてください。

AI ツールがコードの生成や修正に無視できない形で関与した場合は、次のことを行ってください。

1. **入念にレビューする**: 提出したコードの責任は引き続きあなたにあります。自分で書いたコードと同じ注意深さで、AI が生成したコードをレビューし理解してください。
2. **PR で開示する**: PR に AI が生成したコードが含まれる場合は必ず明記してください。PR の説明に注記を追加します。
3. **コミットに印を付ける**: `Co-authored-by:` などのコミットトレーラーで帰属を示してください（他のプロジェクトでは `Assisted-by:` や `Generated-by:` を使うこともあります）。例:

   ```text
   Your commit message here

   Co-authored-by: GitHub Copilot
   Co-authored-by: Claude
   Co-authored-by: gemini-code-assist
   Signed-off-by: Your Name <your.email@example.com>
   ```

AI を活用したコードも、適切なテスト、ドキュメント、スタイルガイドの遵守、入念なレビューといったすべての品質基準を満たす必要があります。帰属表示は、レビュアーが文脈を踏まえて貢献を評価するのに役立ち、プロジェクトの法的な明確さを保ちます。

### PR のタイトルと分類 { #pr-title-and-classification }

レビュー対象となるのは特定の種類の PR のみです。変更の種類を示すため、PR のタイトルには適切な接頭辞を付けてください。次のいずれかを使ってください。

- `[Bugfix]` バグ修正。
- `[CI/Build]` ビルドまたは継続的インテグレーションの改善。
- `[Doc]` ドキュメントの修正・改善。
- `[Model]` 新しいモデルの追加、または既存モデルの改善。タイトルにモデル名を含めてください。
- `[Frontend]` vLLM のフロントエンド（OpenAI API サーバー、`LLM` クラスなど）に関する変更。
- `[Kernel]` CUDA カーネルやその他の計算カーネルに影響する変更。
- `[Core]` vLLM の中核ロジック（`LLMEngine`、`AsyncLLMEngine`、`Scheduler` など）の変更。
- `[Hardware][Vendor]` ハードウェア固有の変更。接頭辞にベンダー名を含めてください（例: `[Hardware][AMD]`）。
- `[Misc]` 上記のカテゴリに当てはまらない PR。
  使用は控えめにしてください。

!!! note
    PR が複数のカテゴリにまたがる場合は、関連するすべての接頭辞を含めてください。

### コード品質 { #code-quality }

PR は次のコード品質基準を満たす必要があります。

- [Google Python スタイルガイド](https://google.github.io/styleguide/pyguide.html)と [Google C++ スタイルガイド](https://google.github.io/styleguide/cppguide.html)に従っています。
- すべての lint チェックに合格すること。
- 将来の貢献者がコードを容易に理解できるよう、十分にドキュメント化されていること。
- プロジェクトが正しく堅牢であり続けるよう、十分なテストを含めること。ユニットテストと結合テストの両方が対象です。
- PR が vLLM のユーザー向けの挙動を変更する場合は、`docs/` にドキュメントを追加してください。vLLM のユーザーが新機能や変更点を理解し活用するのに役立ちます。

### カーネルの追加・変更 { #adding-or-changing-kernels }

カーネルを積極的に開発・変更する際は、ビルド時間を短縮するために[インクリメンタルビルドのワークフロー](./incremental_build.md)の利用を強く推奨します。各カスタムカーネルには、PyTorch に登録するためのスキーマと 1 つ以上の実装が必要です。

- カスタム op は PyTorch のガイドラインに従って登録してください:
  [Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
  および [The Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU)。
- `Tensors` を返すカスタム演算にはメタ関数が必要です。動的な次元を自動的に扱えるよう、メタ関数は Python で実装・登録してください。メタ関数の説明は上記のドキュメントを参照してください。
- 登録した op について、関数の登録とメタ関数をテストするには [torch.library.opcheck()](https://pytorch.org/docs/stable/library.html#torch.library.opcheck) を使ってください。例は `tests/kernels` を参照してください。
- 既存の op の C++ シグネチャを変更する場合は、その変更を反映するようスキーマも更新しなければなりません。
- 新しいカスタム型が必要な場合は、次のドキュメントを参照してください:
  [Custom Class Support in PT2](https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA)。

### 大きな変更に関する注意 { #notes-for-large-changes }

変更はできるだけ簡潔に保ってください。大きなアーキテクチャ変更（カーネル / データ / 設定 / テストを除いて 500 行超）については、技術的な設計と妥当性を議論する GitHub issue（RFC）を期待します。そうでない場合は `rfc-required` のタグを付け、PR を進めない場合があります。

### レビューについて期待できること { #what-to-expect-for-the-reviews }

vLLM チームの目標は *透明性のあるレビューの仕組み* であることです。レビュープロセスを透明かつ効率的にし、貢献者が戸惑ったり不満を感じたりしないようにしたいと考えています。ただし vLLM チームは小規模なため、一部の PR を優先せざるを得ません。レビュープロセスについて期待できることは次のとおりです。

- PR を提出すると、レビュアーが割り当てられます。各レビュアーは自身の専門性と空き状況にもとづいて PR を引き受けます。
- 割り当て後、レビュアーは 2〜3 日ごとに状況を更新します。7 日以内にレビューされない場合は、遠慮なくレビュアーや vLLM チームにメンションしてください。
- レビュー後、変更が必要であればレビュアーが PR に `action-required` ラベルを付けます。貢献者はコメントに対応し、レビュアーにメンションして再レビューを依頼してください。
- すべてのコメントに、妥当な期間内に返答してください。コメントが分かりにくい場合や提案に同意できない場合は、遠慮なく説明を求めたり議論したりしてください。
- 計算資源が限られているため、すべての CI チェックが実行されるわけではありません。PR がマージ可能な状態になったとき、あるいはフル CI の実行が必要なとき、レビュアーが `ready` ラベルを付けます。

### PR の上限とエスカレーション { #pull-request-limits-and-escalation }

vLLM では、書き込み権限のない貢献者に対して GitHub の [pull request limit](https://github.blog/open-source/maintainers/how-pull-request-limits-are-cutting-down-the-noise/) を利用しています。現在の上限はオープンな PR 6 件です。これが善意にもとづく重要な作業を妨げる場合は、コミッターに連絡してバイパスリストへの追加を依頼してください。

重要な貢献について迅速なレビューが必要な場合は、次のアドレスまでメールしてください。

<pr-review-request@vllm.ai>

確認可能な企業または大学のメールアドレスから、次の内容を含めて送ってください。

- 本番環境または研究におけるユースケース
- 直面した問題
- あなたの貢献がそれをどう解決するか

## 謝辞 { #thank-you }

最後に、このガイドラインを読む時間を取っていただき、また vLLM への貢献に関心を持っていただきありがとうございます。皆さんの貢献はすべて、vLLM を誰にとっても素晴らしいツールとコミュニティにするのに役立っています。
