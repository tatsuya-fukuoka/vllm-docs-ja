# Anyscale { #anyscale }

[Anyscale](https://www.anyscale.com) は、Ray の開発者が作ったマネージドのマルチクラウドプラットフォームです。

Anyscale は、AWS・GCP・Azure のアカウント上で Ray クラスタのライフサイクル全体を自動化します。Kubernetes のコントロールプレーンの保守、オートスケーラーの設定、可観測性スタックの管理、[examples/ray_serving/run_cluster.sh](../../../examples/ray_serving/run_cluster.sh) のような補助スクリプトでのヘッドノード・ワーカーノードの手動管理といった運用の手間なしに、オープンソースの Ray の柔軟性を利用できます。

vLLM で大規模言語モデルをサービングする場合、Anyscale は[本番運用に耐える HTTPS エンドポイント](https://docs.anyscale.com/examples/deploy-ray-serve-llms)や[耐障害性のあるバッチ推論ジョブ](https://docs.anyscale.com/examples/ray-data-llm)を素早く用意できます。

## Anyscale 上の本番向け vLLM のクイックスタート { #production-ready-vllm-on-anyscale-quickstarts }

- [オフラインバッチ推論](https://console.anyscale.com/template-preview/llm_batch_inference?utm_source=vllm_docs)
- [vLLM サービスのデプロイ](https://console.anyscale.com/template-preview/llm_serving?utm_source=vllm_docs)
- [データセットのキュレーション](https://console.anyscale.com/template-preview/audio-dataset-curation-llm-judge?utm_source=vllm_docs)
- [LLM のファインチューニング](https://console.anyscale.com/template-preview/entity-recognition-with-llms?utm_source=vllm_docs)
