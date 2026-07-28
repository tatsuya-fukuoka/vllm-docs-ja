# 環境変数 { #environment-variables }

vLLM はシステムの設定に次の環境変数を使用します。

!!! warning
    `VLLM_PORT` と `VLLM_HOST_IP` は vLLM の**内部処理用**のポートと IP を設定するものである点に注意してください。API サーバーのポートと IP ではありません。`--host $VLLM_HOST_IP` や `--port $VLLM_PORT` で API サーバーを起動しても意図どおりには動作しません。

    vLLM 固有の環境変数はほとんどが `VLLM_` で始まります（`CUDA_VISIBLE_DEVICES`、`MAX_JOBS`、`S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` / `S3_ENDPOINT_URL`、`DO_NOT_TRACK`、`NO_COLOR` など、一部の標準的な名前も設定されていれば読み取られます）。**Kubernetes を利用している場合は特に注意してください**。サービス名を `vllm` にしないでください。[Kubernetes はサービスごとに、大文字化したサービス名を接頭辞とする環境変数を設定する](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables)ため、vLLM の環境変数と衝突する可能性があります。

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
