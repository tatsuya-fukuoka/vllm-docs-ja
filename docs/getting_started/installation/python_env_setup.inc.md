<!-- markdownlint-disable MD041 -->
Python 環境の作成・管理には、非常に高速な環境マネージャーである [uv](https://docs.astral.sh/uv/) の使用をおすすめします。`uv` のインストール方法は[ドキュメント](https://docs.astral.sh/uv/#getting-started)を参照してください。`uv` をインストールしたら、次のコマンドで新しい Python 環境を作成できます。

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```
