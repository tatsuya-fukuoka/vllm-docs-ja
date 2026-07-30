# vLLM wheel の nightly ビルド { #nightly-builds-of-vllm-wheels }

vLLM は `https://wheels.vllm.ai` にコミットごとの wheel リポジトリ（一般に「nightly」と呼ばれます）を用意しており、`v0.5.3` 以降の `main` ブランチのすべてのコミットについてビルド済み wheel を提供しています。このドキュメントでは、nightly wheel のインデックスの仕組みを説明します。

## CI におけるビルドとアップロードの流れ { #build-and-upload-process-on-ci }

### wheel のビルド { #wheel-building }

wheel は、PR が main ブランチにマージされたあと `Release` パイプライン（`.buildkite/release-pipeline.yaml`）でビルドされます。通常のビルドでは、x86_64 と aarch64 向けの CUDA 13.0 wheel が生成されます。追加の wheel バリアントと ROCm ビルドは必要に応じてブロック解除でき、`NIGHTLY=1` の場合は自動的に実行されます。

- **バックエンドのバリアント**: `cpu` と `cuXXX`（`cu129`、`cu130` など）
- **アーキテクチャのバリアント**: `x86_64` と `aarch64`

各ビルドステップでは次のことを行います。

1. Docker コンテナ内で wheel をビルドする。
2. PEP 600 に準拠するため、wheel のファイル名を正しい manylinux タグ（現在は `manylinux_2_28`）に変更する。
3. wheel を S3 バケット `vllm-wheels` の `/{commit_hash}/` にアップロードする。

### インデックスの生成 { #index-generation }

各 wheel のアップロード後、`.buildkite/scripts/upload-wheels.sh` スクリプトは次のことを行います。

1. **既存のすべての wheel を一覧表示**する（S3 上のそのコミットのディレクトリから）
2. `.buildkite/scripts/generate-nightly-index.py` を使って**インデックスを生成**する:
    - wheel のファイル名を解析してメタデータ（バージョン、バリアント、プラットフォームタグ）を抽出する。
    - PyPI 互換のための HTML インデックスファイル（`index.html`）を作成する。
    - 機械可読な `metadata.json` ファイルを生成する。
3. **インデックスを複数の場所にアップロード**する（既存のものは上書き）:
    - `/{commit_hash}/` - コミット単位でアクセスできるよう常にアップロードされます。
    - `/nightly/` - `main` ブランチのコミットの場合のみ（PR は対象外）。
    - `/{version}/` - リリース用 wheel（バージョンに `dev` を含まないもの）の場合のみ。

!!! tip "同時ビルドへの対応"
    インデックス生成スクリプトは、インデックスを生成する前に必ずそのコミットのディレクトリにある
    すべての wheel を一覧表示するため、複数のバリアントが同時にビルドされても競合状態を避けられます。

## ディレクトリ構造 { #directory-structure }

S3 バケットの構造は次のパターンに従います。

```text
s3://vllm-wheels/
├── {commit_hash}/              # Commit-specific wheels and indices
│   ├── vllm-*.whl              # All wheel files
│   ├── index.html              # Project list (default variant)
│   ├── vllm/
│   │   ├── index.html          # Package index (default variant)
│   │   └── metadata.json       # Metadata (default variant)
│   ├── cu129/                  # Variant subdirectory
│   │   ├── index.html          # Project list (cu129 variant)
│   │   └── vllm/
│   │       ├── index.html      # Package index (cu129 variant)
│   │       └── metadata.json   # Metadata (cu129 variant)
│   ├── cu130/                  # Variant subdirectory
│   ├── cpu/                    # Variant subdirectory
│   └── .../                    # More variant subdirectories
├── nightly/                    # Latest main branch wheels (mirror of latest commit)
└── {version}/                  # Release version indices (e.g., 0.11.2)
```

ビルドされた wheel はすべて `/{commit_hash}/` に格納され、さまざまなインデックスが生成されてそれらを参照します。これにより wheel ファイルの重複を避けられます。

たとえば、次のような URL を指定して異なるインデックスを使えます。

- `https://wheels.vllm.ai/nightly/cu130` — CUDA 13.0 でビルドされた最新の main ブランチの wheel
- `https://wheels.vllm.ai/{commit_hash}` — 特定コミットでビルドされた wheel（既定のバリアント）
- `https://wheels.vllm.ai/0.12.0/cpu` — CPU バリアント向けにビルドされた 0.12.0 リリースの wheel

すべてのコミットにすべてのバリアントが存在するわけではない点に注意してください。利用可能なバリアントは、たとえば cu130 が cu131 に変わるなど、時間とともに変化します。

### バリアントの構成 { #variant-organization }

インデックスはバリアントごとに整理されます。

- **既定のバリアント**: バリアントの接尾辞がない wheel（つまり現在の `VLLM_MAIN_CUDA_VERSION` でビルドされたもの）はルートに置かれます。
- **バリアントのサブディレクトリ**: バリアントの接尾辞（`+cu130`、`.cpu` など）を持つ wheel はサブディレクトリに整理されます。
- **既定バリアントのエイリアス**: 一貫性と利便性のため、既定のバリアントにはエイリアス（現時点では `cu129` など）を設けられます。

バリアントは wheel のファイル名から抽出されます（[ファイル名の規約](https://packaging.python.org/en/latest/specifications/binary-distribution-format/#file-name-convention)を参照）。

- バリアントはローカルバージョン識別子にエンコードされます（`+cu129` や `dev<N>+g<hash>.cu130` など）。
- 例:
    - `vllm-0.11.2.dev278+gdbc3d9991-cp38-abi3-manylinux1_x86_64.whl` → 既定のバリアント
    - `vllm-0.10.2rc2+cu129-cp38-abi3-manylinux2014_aarch64.whl` → `cu129` バリアント
    - `vllm-0.11.1rc8.dev14+gaa384b3c0.cu130-cp38-abi3-manylinux1_x86_64.whl` → `cu130` バリアント

## インデックス生成の詳細 { #index-generation-details }

`generate-nightly-index.py` スクリプトは次のことを行います。

1. 正規表現で **wheel のファイル名を解析**し、次を抽出します。
    - パッケージ名
    - バージョン（バリアントを抽出したもの）
    - Python タグ、ABI タグ、プラットフォームタグ
    - ビルドタグ（存在する場合）
2. **wheel をバリアントごと、次いでパッケージ名ごとにグループ化**します。
    - 現時点でビルドされるのは `vllm` のみですが、この構造は将来的な複数パッケージにも対応できます。
3. **HTML インデックスを生成**します（[Simple repository API](https://packaging.python.org/en/latest/specifications/simple-repository-api/#simple-repository-api) に準拠）。
    - トップレベルの `index.html`: すべてのパッケージとバリアントのサブディレクトリを列挙します
    - パッケージレベルの `index.html`: そのパッケージのすべての wheel ファイルを列挙します
    - 可搬性のため、wheel ファイルへは相対パスを使います
4. **metadata.json を生成**します。
    - すべての wheel メタデータを含む機械可読な JSON です
    - wheel ファイルへの URL エンコード済み相対パスを持つ `path` フィールドを含みます
    - Python のみのビルド時に、互換性のあるビルド済み wheel を見つけるために `setup.py` が使用します

### AWS サービスに関する特別な対応 { #special-handling-for-aws-services }

wheel とインデックスは AWS S3 に直接格納され、S3 バケットの前段の CDN として AWS CloudFront を使っています。

S3 には適切なディレクトリ一覧の機能がないため、PyPI 互換の simple repository API の挙動を実現するために、次のことを行う CloudFront Function をデプロイしています。

- `/` で終わらず、ファイルのように見えない URL（最後のパスセグメントにドット `.` を含まないもの）は、末尾に `/` を付けた同じ URL にリダイレクトする
- `/` で終わる URL には `/index.html` を付加する

たとえば、次のリクエストは以下のように処理されます。

- `/nightly` -> `/nightly/index.html`
- `/nightly/cu130/` -> `/nightly/cu130/index.html`
- `/nightly/index.html` や `/nightly/vllm.whl` -> そのまま

!!! note "AWS S3 のファイル名エスケープ"

    S3 はアップロード時に、[命名規則](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html)に従ってファイル名を自動的にエスケープします。vLLM への直接的な影響は、ファイル名中の `+` が `%2B` に変換される点です。インデックス生成スクリプトでは、HTML インデックスと JSON メタデータを生成する際にファイル名を適切にエスケープするよう特別な配慮をしており、URL が正しくそのまま使えるようにしています。

## `setup.py` におけるビルド済み wheel の利用 { #precompiled-wheels-usage }

`VLLM_USE_PRECOMPILED=1` を指定して vLLM をインストールすると、`setup.py` スクリプトは次のことを行います。

1. `precompiled_wheel_utils.determine_wheel_url()` によって **wheel の場所を決定**します。
    - 環境変数 `VLLM_PRECOMPILED_WHEEL_LOCATION`（ユーザーが指定した URL / パス）が常に優先され、他のステップはすべてスキップされます。
    - `VLLM_MAIN_CUDA_VERSION` からバリアントを決定します（環境変数 `VLLM_PRECOMPILED_WHEEL_VARIANT` で上書き可能）。フォールバックとして既定のバリアントも試されます。
    - このブランチの _ベースコミット_（後述）を決定します（環境変数 `VLLM_PRECOMPILED_WHEEL_COMMIT` で上書き可能）。
2. `https://wheels.vllm.ai/{commit}/vllm/metadata.json`（既定のバリアントの場合）または `https://wheels.vllm.ai/{commit}/{variant}/vllm/metadata.json`（特定のバリアントの場合）から**メタデータを取得**します。
3. 次にもとづいて**互換性のある wheel を選択**します。
    - パッケージ名（`vllm`）
    - プラットフォームタグ（アーキテクチャの一致）
4. wheel からビルド済みの成果物を**ダウンロードして展開**します。
    - ネイティブ拡張モジュール（`.so` ファイル）
    - `vllm-rs` の Rust フロントエンドのバイナリ
    - Flash Attention の Python モジュールと Triton / FlashMLA の Python ファイル
5. 展開したファイルをインストールに含めるため **package_data にパッチを当てます**。

!!! note "ベースコミットとは"

    ベースコミットは、現在のブランチと上流の `main` の merge-base を求めることで決定され、
    ソースコードとビルド済みバイナリの互換性を担保します。

_注: ビルド済み wheel を使う前に、ネイティブコード（C++ や CUDA など）に変更がないことを確認するのは利用者の責任です。_

## 実装ファイル { #implementation-files }

nightly wheel の仕組みに関わる主要なファイルは次のとおりです。

- **`.buildkite/release-pipeline.yaml`**: wheel をビルドする CI パイプライン
- **`.buildkite/scripts/upload-wheels.sh`**: wheel をアップロードし、インデックスを生成するスクリプト
- **`.buildkite/scripts/generate-nightly-index.py`**: PyPI 互換のインデックスを生成する Python スクリプト
- **`setup.py`**: ビルド済み wheel の取得と利用を行う `precompiled_wheel_utils` クラスを含みます
