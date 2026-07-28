# 分散デプロイのトラブルシューティング { #troubleshooting-distributed-deployments }

一般的なトラブルシューティングは[トラブルシューティング](../usage/troubleshooting.md)を参照してください。

## ノード間の GPU 通信を確認する { #verify-inter-node-gpu-communication }

Ray クラスタを起動したら、ノードをまたぐ GPU 間通信を確認してください。適切な設定は簡単ではありません。詳細は[確認用スクリプト](../usage/troubleshooting.md#incorrect-hardwaredriver)を参照してください。通信の設定に追加の環境変数が必要な場合は、[examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) に `-e NCCL_SOCKET_IFNAME=eth0` のように追記してください。クラスタ作成時に環境変数を設定すると、すべてのノードに伝播するため推奨されます。一方、シェルで設定した環境変数はローカルのノードにしか影響しません。詳細は <https://github.com/vllm-project/vllm/issues/6803> を参照してください。

## No available node types can fulfill resource request { #no-available-node-types-can-fulfill-resource-request }

`Error: No available node types can fulfill resource request` というエラーは、クラスタに十分な GPU があっても発生することがあります。ノードが複数の IP アドレスを持ち、vLLM が正しいものを選択できない場合によく起こります。[examples/ray_serving/run_cluster.sh](../../examples/ray_serving/run_cluster.sh) で `VLLM_HOST_IP` を設定し（ノードごとに異なる値）、vLLM と Ray が同じ IP アドレスを使うようにしてください。選択された IP アドレスは `ray status` と `ray list nodes` で確認できます。詳細は <https://github.com/vllm-project/vllm/issues/7815> を参照してください。

## Ray の可観測性 { #ray-observability }

分散システムは規模が大きく複雑なため、デバッグが難しくなりがちです。Ray は、Ray のアプリケーションとクラスタを監視・デバッグ・最適化するための一連のツールを提供しています。Ray の可観測性については [Ray 公式の可観測性ドキュメント](https://docs.ray.io/en/latest/ray-observability/index.html)（英語）を参照してください。Ray アプリケーションのデバッグについては [Ray Debugging Guide](https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/index.html)（英語）を参照してください。Kubernetes クラスタのトラブルシューティングについては
[KubeRay 公式のトラブルシューティングガイド](https://docs.ray.io/en/latest/serve/advanced-guides/multi-node-gpu-troubleshooting.html)（英語）を参照してください。
