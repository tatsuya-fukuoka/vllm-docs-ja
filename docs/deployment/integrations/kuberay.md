# KubeRay { #kuberay }

[KubeRay](https://github.com/ray-project/kuberay) は、Ray クラスタ上で vLLM のワークロードを動かすための Kubernetes ネイティブな方法を提供します。
Ray クラスタを YAML で宣言でき、オペレーターが Pod のスケジューリング、ネットワークの設定、再起動、ブルーグリーンデプロイを、使い慣れた Kubernetes の作法のまま処理します。

## 手動スクリプトではなく KubeRay を使う理由 { #why-kuberay-instead-of-manual-scripts }

| 機能 | 手動スクリプト | KubeRay |
| ------- | --------------------------------------------------------- | ------- |
| クラスタの初期構築 | 各ノードに手動で SSH してスクリプトを実行 | 1 コマンドでクラスタ全体を作成・更新: `kubectl apply -f cluster.yaml` |
| オートスケーリング | 手動 | クラスタサイズの調整のため CRD を自動的にパッチ |
| アップグレード | 手動で破棄して作り直す | ブルーグリーンデプロイによる更新に対応 |
| 宣言的な設定 | bash のフラグと環境変数 | GitOps と相性の良い YAML の CRD (RayCluster / RayService) |

KubeRay を使うと運用負荷が下がり、Ray + vLLM を既存の Kubernetes のワークフロー（CI/CD、Secret、StorageClass など）に統合しやすくなります。

## さらに詳しく { #learn-more }

* [「Serve a Large Language Model using Ray Serve LLM on Kubernetes」](https://docs.ray.io/en/master/cluster/kubernetes/examples/rayserve-llm-example.html) - vLLM・KubeRay・Ray Serve を使ってモデルをサービングするエンドツーエンドの例（英語）。
* [KubeRay のドキュメント](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)（英語）
