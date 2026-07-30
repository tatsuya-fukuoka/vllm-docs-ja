# Dockerfile { #dockerfile }

vLLM で OpenAI 互換サーバーを動かすためのイメージを構築する [docker/Dockerfile](../../../docker/Dockerfile) を提供しています。
Docker でのデプロイの詳細は[こちら](../../deployment/docker.md)を参照してください。

以下は、マルチステージの Dockerfile を図示したものです。ビルドグラフには次のノードが含まれます。

- すべてのビルドステージ
- 既定のビルドターゲット（グレーで強調）
- 外部イメージ（破線の枠）

ビルドグラフの辺は次を表します。

- `FROM ...` の依存関係（実線と塗りつぶしの矢印）

- `COPY --from=...` の依存関係（破線と白抜きの矢印）

- `RUN --mount=(.\*)from=...` の依存関係（点線と白抜きのひし形の矢印）

  > <figure markdown="span">
  >   ![](https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/docs/assets/contributing/dockerfile-stages-dependency.png){ align="center" alt="query" width="100%" }
  > </figure>
  >
  > 作成に使用したツール: <https://github.com/patrickhoefler/dockerfilegraph>
  >
  > ビルドグラフを再生成するコマンド（Dockerfile がある **vLLM リポジトリの \`root\` ディレクトリ**で実行してください）:
  >
  > ```bash
  > dockerfilegraph \
  >   -o png \
  >   --legend \
  >   --dpi 200 \
  >   --max-label-length 50 \
  >   --filename docker/Dockerfile
  > ```
  >
  > Docker イメージで直接実行する場合は次のようにします。
  >
  > ```bash
  > docker run \
  >    --rm \
  >    --user "$(id -u):$(id -g)" \
  >    --workdir /workspace \
  >    --volume "$(pwd)":/workspace \
  >    ghcr.io/patrickhoefler/dockerfilegraph:alpine \
  >    --output png \
  >    --dpi 200 \
  >    --max-label-length 50 \
  >    --filename docker/Dockerfile \
  >    --legend
  > ```
  >
  >（別のファイルを対象にする場合は、`--filename` フラグに別の引数を渡してください。）
