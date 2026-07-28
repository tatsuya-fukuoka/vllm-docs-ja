# 利用統計の収集 { #usage-stats-collection }

vLLM は既定で匿名の利用データを収集しています。これは、どのハードウェアやモデル構成が広く使われているかを開発チームが把握し、もっとも一般的なワークロードに優先的に取り組めるようにするためです。収集内容は公開されており、機微な情報は含まれません。

データの一部は、整形・集計したうえでコミュニティのために公開されます。たとえば 2024 年の利用レポートは[こちら](https://2024.vllm.ai)（英語）で見られます。

## どのようなデータが収集されるか { #what-data-is-collected }

最新バージョンの vLLM が収集するデータの一覧はこちらにあります: [vllm/usage/usage_lib.py](../../vllm/usage/usage_lib.py)

v0.4.0 時点の例を示します。

??? console "Output"

    ```json
    {
      "uuid": "fbe880e9-084d-4cab-a395-8984c50f1109",
      "provider": "GCP",
      "num_cpu": 24,
      "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
      "cpu_family_model_stepping": "6,85,7",
      "total_memory": 101261135872,
      "architecture": "x86_64",
      "platform": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
      "gpu_count": 2,
      "gpu_type": "NVIDIA L4",
      "gpu_memory_per_device": 23580639232,
      "model_architecture": "OPTForCausalLM",
      "vllm_version": "0.3.2+cu123",
      "context": "LLM_CLASS",
      "log_time": 1711663373492490000,
      "source": "production",
      "dtype": "torch.float16",
      "tensor_parallel_size": 1,
      "block_size": 16,
      "gpu_memory_utilization": 0.9,
      "quantization": null,
      "kv_cache_dtype": "auto",
      "enable_lora": false,
      "enable_prefix_caching": false,
      "enforce_eager": false,
      "disable_custom_all_reduce": true
    }
    ```

収集されたデータは次のコマンドで確認できます。

```bash
tail ~/.config/vllm/usage_stats.json
```

## 収集を停止する { #opting-out }

環境変数 `VLLM_NO_USAGE_STATS` または `DO_NOT_TRACK` を設定するか、`~/.config/vllm/do_not_track` ファイルを作成すると、利用統計の収集を停止できます。

```bash
# Any of the following methods can disable usage stats collection
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
mkdir -p ~/.config/vllm && touch ~/.config/vllm/do_not_track
```
