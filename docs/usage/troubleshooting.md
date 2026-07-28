# トラブルシューティング { #troubleshooting }

このドキュメントでは、問題が起きたときに試せる切り分けの方法を紹介します。バグを見つけたと思われる場合は、まず[既存の Issue を検索](https://github.com/vllm-project/vllm/issues?q=is%3Aissue)して、すでに報告されていないか確認してください。見つからない場合は、関連する情報をできるだけ添えて[新しい Issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)してください。

!!! note
    問題のデバッグが終わったら、設定したデバッグ用の環境変数を必ず解除するか、新しいシェルを開いて、残ったデバッグ設定の影響を受けないようにしてください。そうしないと、デバッグ機能が有効なままでシステムが遅くなることがあります。

## モデルのダウンロードで固まる { #hangs-downloading-a-model }

モデルがまだディスクにダウンロードされていない場合、vLLM はインターネットからダウンロードします。これには時間がかかり、回線状況にも左右されます。
先に [huggingface-cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) でモデルをダウンロードし、ローカルパスを vLLM に渡すことをおすすめします。これにより問題を切り分けられます。

## ディスクからのモデル読み込みで固まる { #hangs-loading-a-model-from-disk }

モデルが大きいと、ディスクからの読み込みに長い時間がかかります。モデルの保存場所に注意してください。クラスタによってはノード間で共有されるファイルシステム（分散ファイルシステムやネットワークファイルシステムなど）が使われており、速度が遅い場合があります。
モデルはローカルディスクに置くほうが望ましいです。また、CPU メモリの使用量も確認してください。モデルが大きすぎると CPU メモリを大量に消費し、ディスクとメモリの間でスワップが頻発して OS 全体が遅くなることがあります。

!!! note
    モデルのダウンロードと読み込みの問題を切り分けるには、`--load-format dummy` 引数でモデルの重みの読み込みをスキップできます。これにより、ダウンロードや読み込みがボトルネックかどうかを確認できます。

## メモリ不足 (OOM) { #out-of-memory }

モデルが大きすぎて 1 つの GPU に収まらない場合、メモリ不足 (OOM) エラーが発生します。メモリ使用量を減らすために[これらのオプション](../configuration/conserving_memory.md)の利用を検討してください。

## 生成品質が変わった { #generation-quality-changed }

v0.8.0 で、既定のサンプリングパラメータの取得元が <https://github.com/vllm-project/vllm/pull/12622> により変更されました。v0.8.0 より前は vLLM が定めた中立的な既定値が使われていましたが、v0.8.0 以降はモデル作成者が提供する `generation_config.json` の値が使われます。

多くの場合、モデル作成者は自分のモデルに最適なサンプリングパラメータを把握しているため、これは応答の品質向上につながります。ただし、モデル作成者が提供する既定値が性能低下を招く場合もあります。

これが原因かどうかは、オンラインなら `--generation-config vllm`、オフラインなら `generation_config="vllm"` で以前の既定値を試すと確認できます。これで生成品質が改善する場合は、vLLM の既定値を使い続けたうえで、<https://huggingface.co> でモデル作成者に `generation_config.json` の既定値をより良い品質になるよう更新するよう働きかけることをおすすめします。

## ログを増やす { #enable-more-logging }

他の方法で解決しない場合、vLLM のインスタンスがどこかで停止している可能性があります。次の環境変数がデバッグに役立ちます。

- `export VLLM_LOGGING_LEVEL=DEBUG`: ログ出力を増やします。
- `export VLLM_LOG_STATS_INTERVAL=1.`: 実行中キュー・待機キュー・キャッシュヒット状況を追跡するため、統計ログをより頻繁に出力します。
- `export CUDA_LAUNCH_BLOCKING=1`: どの CUDA カーネルが問題を起こしているかを特定します。
- `export NCCL_DEBUG=TRACE`: NCCL のログ出力を増やします。
- `export VLLM_TRACE_FUNCTION=1`: すべての関数呼び出しをログファイルに記録し、どの関数がクラッシュ・停止しているかを調べます。（注意: このフラグはトークン生成を **100 倍以上**遅くします。どうしても必要な場合以外は使わないでください。）

## ブレークポイント { #breakpoints }

vLLM のコードベースでは、サブプロセス内で実行される箇所に通常の `pdb` のブレークポイントを設定しても機能しないことがあります。次のような出力になります。

``` text
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 100, in trace_dispatch
    return self.dispatch_line(frame)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/uv/cpython-3.12.11-linux-x86_64-gnu/lib/python3.12/bdb.py", line 125, in dispatch_line
    if self.quitting: raise BdbQuit
                      ^^^^^^^^^^^^^
bdb.BdbQuit
```

1 つの解決策は [forked-pdb](https://github.com/Lightning-AI/forked-pdb) を使うことです。`pip install fpdb` でインストールし、次のようにブレークポイントを設定します。

``` python
__import__('fpdb').ForkedPdb().set_trace()
```

もう 1 つの方法は、環境変数 `VLLM_ENABLE_V1_MULTIPROCESSING` でマルチプロセスを完全に無効化することです。
スケジューラが同じプロセス内に留まるため、標準の `pdb` のブレークポイントが使えます。

``` python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
```

## ネットワーク設定が正しくない { #incorrect-network-setup }

ネットワーク構成が複雑な場合、vLLM のインスタンスが正しい IP アドレスを取得できないことがあります。`DEBUG 06-10 21:32:17 parallel_state.py:88] world_size=8 rank=0 local_rank=0 distributed_init_method=tcp://xxx.xxx.xxx.xxx:54641 backend=nccl` のようなログを確認し、IP アドレスが正しいかを見てください。
正しくない場合は、環境変数 `export VLLM_HOST_IP=<your_ip_address>` で上書きします。

その IP アドレスに対応するネットワークインターフェイスを指定するため、`export NCCL_SOCKET_IFNAME=<your_network_interface>` と `export GLOO_SOCKET_IFNAME=<your_network_interface>` の設定も必要になる場合があります。

## `self.graph.replay()` 付近でのエラー { #error-near-selfgraphreplay }

vLLM がクラッシュし、エラートレースが `vllm/worker/model_runner.py` の `self.graph.replay()` 付近を指している場合、これは CUDAGraph 内部で発生した CUDA エラーです。
原因となっている CUDA 演算を特定するには、コマンドラインに `--enforce-eager` を追加するか、[`LLM`](https://docs.vllm.ai/en/v0.26.0/api/vllm/#vllm.LLM) クラスに `enforce_eager=True` を指定して CUDAGraph の最適化を無効にし、問題の演算を切り分けてください。

## ハードウェア・ドライバの問題 { #incorrect-hardwaredriver }

GPU / CPU 間の通信が確立できない場合は、次の Python スクリプトと手順を使って、通信が正しく機能しているか確認できます。

??? code

    ```python
    # Test PyTorch NCCL
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.accelerator.device_count()
    torch.accelerator.set_device_index(local_rank)
    data = torch.FloatTensor([1,] * 128).to("cuda")
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    torch.accelerator.synchronize()
    value = data.mean().item()
    world_size = dist.get_world_size()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("PyTorch NCCL is successful!")

    # Test PyTorch GLOO
    gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    cpu_data = torch.FloatTensor([1,] * 128)
    dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
    value = cpu_data.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("PyTorch GLOO is successful!")

    if world_size <= 1:
        exit()

    # Test vLLM NCCL, with cuda graph
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
    # pynccl is enabled by default for 0.6.5+,
    # but for 0.6.4 and below, we need to enable it manually.
    # keep the code for backward compatibility when because people
    # prefer to read the latest documentation.
    pynccl.disabled = False

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        data.fill_(1)
        out = pynccl.all_reduce(data, stream=s)
        value = out.mean().item()
        assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL is successful!")

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=g, stream=s):
        out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

    data.fill_(1)
    g.replay()
    torch.cuda.current_stream().synchronize()
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

    print("vLLM NCCL with cuda graph is successful!")

    dist.destroy_process_group(gloo_group)
    dist.destroy_process_group()
    ```

単一ノードでテストする場合は、`--nproc-per-node` を使用したい GPU の数に合わせてください。

```bash
NCCL_DEBUG=TRACE torchrun --nproc-per-node=<number-of-GPUs> test.py
```

複数ノードでテストする場合は、環境に合わせて `--nproc-per-node` と `--nnodes` を調整し、`MASTER_ADDR` にはすべてのノードから到達できるマスターノードの IP アドレスとポート（例: `10.0.0.1:29400`）を設定して、次を実行します。

```bash
NCCL_DEBUG=TRACE torchrun --nnodes 2 \
    --nproc-per-node=2 \
    --rdzv_backend=static \
    --rdzv_endpoint=$MASTER_ADDR \
    --node-rank $NODE_RANK test.py
```

`MASTER_ADDR` には、すべてのノードから到達できるマスターノードの IP アドレスとポート（例: `10.0.0.1:29400`）を設定します。`NODE_RANK` はマスターノードで `0`、ワーカーで `1`、`2`、... とします。`--nproc-per-node` と `--nnodes` は環境に合わせて調整してください。

!!! note
    `c10d` ではなく `--rdzv_backend=static` を使っています。これは、複数ノード構成で `c10d` のランデブーバックエンドが DNS 解決エラーで失敗することがあるためです（[pytorch/pytorch#85300](https://github.com/pytorch/pytorch/issues/85300) を参照）。`static` バックエンドはノードのランクを明示的に指定させることで、この問題を回避します。

スクリプトが正常に実行されると、`sanity check is successful!` というメッセージが表示されます。

テストスクリプトが停止したりクラッシュしたりする場合は、通常ハードウェアまたはドライバに何らかの問題があります。システム管理者やハードウェアベンダーに問い合わせてください。よくある回避策として、`export NCCL_P2P_DISABLE=1` など NCCL の環境変数を調整して改善するか試せます。詳細は [NCCL のドキュメント](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)（英語）を参照してください。これらの環境変数はシステムの性能に影響する可能性があるため、あくまで一時的な回避策として使用してください。最善の解決策は、テストスクリプトが正常に動作するようハードウェアやドライバを修正することです。

## Python のマルチプロセス { #python-multiprocessing }

### `RuntimeError` 例外 { #runtimeerror-exception }

ログに次のような警告が出ている場合、

```console
WARNING 12-11 14:50:37 multiproc_worker_utils.py:281] CUDA was previously
    initialized. We must use the `spawn` multiprocessing start method. Setting
    VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. See
    https://docs.vllm.ai/en/latest/usage/troubleshooting.html#python-multiprocessing
    for more information.
```

あるいは Python から次のようなエラーが出ている場合、

??? console "Logs"

    ```console
    RuntimeErrまたは、
            An attempt has been made to start a new process before the
            current process has finished its bootstrapping phase.

            This probably means that you are not using fork to start your
            child processes and you have forgotten to use the proper idiom
            in the main module:

                if __name__ == '__main__':
                    freeze_support()
                    ...

            The "freeze_support()" line can be omitted if the program
            is not going to be frozen to produce an executable.

            To fix this issue, refer to the "Safe importing of main module"
            section in https://docs.python.org/3/library/multiprocessing.html
    ```

`vllm` の利用を `if __name__ == '__main__':` ブロックで囲むように Python コードを修正する必要があります。
たとえば、次のコードの代わりに、

```python
import vllm

llm = vllm.LLM(...)
```

次のように書いてください。

```python
if __name__ == '__main__':
    import vllm

    llm = vllm.LLM(...)
```

## `torch.compile` のエラー { #torchcompile-error }

vLLM は性能向上のためにモデルの最適化を `torch.compile` に大きく依存しており、そのため `torch.compile` の機能と `triton` ライブラリに依存関係があります。既定では、モデル内の[一部の関数を最適化](https://github.com/vllm-project/vllm/pull/10406)するために `torch.compile` を使用します。vLLM を実行する前に、次のスクリプトで `torch.compile` が期待どおり動作するか確認できます。

??? code

    ```python
    import torch

    @torch.compile
    def f(x):
        # a simple function to test torch.compile
        x = x + 1
        x = x * 2
        x = x.sin()
        return x

    x = torch.randn(4, 4).cuda()
    print(f(x))
    ```

`torch/_inductor` ディレクトリ由来のエラーが出る場合、たいていは使用中の PyTorch のバージョンと互換性のない独自の `triton` ライブラリが入っています。例として <https://github.com/vllm-project/vllm/issues/12219> を参照してください。

## モデルの検査に失敗する { #model-failed-to-be-inspected }

次のようなエラーが出る場合、

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] failed to be inspected. Please check the logs for more details.
```

vLLM がモデルのファイルを import できなかったことを意味します。
通常は依存関係の不足や、vLLM のビルドに含まれるバイナリが古いことが原因です。
ログをよく読んで根本原因を特定してください。

## モデルがサポートされていない { #model-not-supported }

次のようなエラーが出る場合、

```text
Traceback (most recent call last):
...
  File "vllm/model_executor/models/registry.py", line xxx, in inspect_model_cls
    for arch in architectures:
TypeError: 'NoneType' object is not iterable
```

または、

```text
  File "vllm/model_executor/models/registry.py", line xxx, in _raise_for_unsupported
    raise ValueError(
ValueError: Model architectures ['<arch>'] are not supported for now. Supported architectures: [...]
```

しかしそのモデルが[対応モデルの一覧](../models/supported_models.md)に含まれていることが確かな場合、vLLM のモデル解決に問題がある可能性があります。その場合は[こちらの手順](../configuration/model_resolution.md)に従って、モデルに対する vLLM の実装を明示的に指定してください。

## デバイス種別の推定に失敗する { #failed-to-infer-device-type }

`RuntimeError: Failed to infer device type` のようなエラーが出る場合、vLLM が実行環境のデバイス種別を推定できなかったことを意味します。vLLM がどのようにデバイス種別を推定しているか、なぜ期待どおりに動かないかは[コード](../../vllm/platforms/__init__.py)で確認できます。[この PR](https://github.com/vllm-project/vllm/pull/14195) 以降は、環境変数 `VLLM_LOGGING_LEVEL=DEBUG` を設定して、より詳細なログを確認することもできます。

## NCCL エラー: `ncclCommInitRank` 中の unhandled system error { #nccl-error-unhandled-system-error-during-nccl-comm-init-rank }

複数ノードにまたがる分散サービングで GPUDirect RDMA を使用していて、`NCCL_DEBUG=INFO` を設定しても明確なエラーメッセージが出ないまま `ncclCommInitRank` でエラーになる場合、次のような出力になります。

```text
Error executing method 'init_device'. This might cause deadlock in distributed execution.
Traceback (most recent call last):
...
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl.py", line 99, in __init__
     self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl_wrapper.py", line 277, in ncclCommInitRank
     self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm),
   File "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/pynccl_wrapper.py", line 256, in NCCL_CHECK
     raise RuntimeError(f"NCCL error: {error_str}")
 RuntimeError: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details)
...
```

これは vLLM が NCCL のコミュニケータを初期化できなかったことを示しており、Linux の `IPC_LOCK` ケーパビリティが付与されていない、または `/dev/shm` がマウントされていないことが原因の可能性があります。GPUDirect RDMA 向けの適切な環境設定については [Enabling GPUDirect RDMA](../serving/parallelism_scaling.md#enabling-gpudirect-rdma) を参照してください。

## CUDA エラー: the provided PTX was compiled with an unsupported toolchain { #cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain }

`RuntimeError: CUDA error: the provided PTX was compiled with an unsupported toolchain` のようなエラーが出る場合、vLLM の wheel に含まれる CUDA PTX が、システムでサポートされていないツールチェーンでコンパイルされていることを意味します。`RuntimeError: The NVIDIA driver on your system is too old` というエラーの場合も、このセクションが該当します。

公開されている vLLM の wheel は特定バージョンの CUDA ツールキットでコンパイルされており、それより古い CUDA ドライバでは動作しないことがあります。詳細は [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)（英語）を参照してください。**この機能は一部のプロフェッショナル向け・データセンター向け NVIDIA GPU でのみサポートされています。**

vLLM 公式の Docker イメージを使っている場合は、`docker run` コマンドに `-e VLLM_ENABLE_CUDA_COMPATIBILITY=1` を追加すると解決できます。これにより、プリインストールされた CUDA 前方互換ライブラリが有効になります。

Docker の外で vLLM を実行している場合は、[CUDA リポジトリ](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)を有効にしたうえで、パッケージマネージャから `cuda-compat` パッケージをインストールします。たとえば Ubuntu では `sudo apt-get install cuda-compat-12-9` を実行し、`export VLLM_ENABLE_CUDA_COMPATIBILITY=1` と `export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.9/compat"` を設定します。

Conda では `conda-forge::cuda-compat` パッケージをインストールし（例: `conda install -c conda-forge cuda-compat=12.9`）、環境を有効化したあとに `export VLLM_ENABLE_CUDA_COMPATIBILITY=1` と `export VLLM_CUDA_COMPATIBILITY_PATH="${CONDA_PREFIX}/cuda-compat"` を設定します。

設定が有効かどうかは、vLLM 経由で CUDA を初期化する最小限の Python スクリプトで確認できます。

```bash
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="/usr/local/cuda-12.9/compat"

python3 - << 'EOF'
import vllm
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.accelerator.device_count()}")
EOF
```

ここでは例として CUDA 12.9 を使っていますが、vLLM の既定の CUDA バージョンが上がった場合は、より新しい cuda-compat パッケージをインストールしてください。

## ptxas fatal: Value 'sm_110a' is not defined for option 'gpu-name' { #ptxas-fatal-value-sm110a-is-not-defined-for-option-gpu-name }

CUDA 13 で triton のカーネルを使うと、`ptxas fatal: Value 'sm_110a' is not defined for option 'gpu-name'` のようなエラーが出ることがあります。

```text
(EngineCore_0 pid=9492) triton.runtime.errors.PTXASError: PTXAS error: Internal Triton PTX codegen error
(EngineCore_0 pid=9492) `ptxas` stderr:
(EngineCore_0 pid=9492) ptxas fatal   : Value 'sm_110a' is not defined for option 'gpu-name'
(EngineCore_0 pid=9492) 
(EngineCore_0 pid=9492) Repro command: /home/jetson/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas -lineinfo -v --gpu-name=sm_110a /tmp/tmp95oy_b9d.ptx -o /tmp/tmp95oy_b9d.ptx.o
(EngineCore_0 pid=9492) 
    outputs = self.engine_core.get_output()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jetson/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 668, in get_output
    raise self._format_exception(outputs) from None
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

これは triton に同梱されている ptxas が使用中のデバイスと互換性がないことを意味します。環境変数 `TRITON_PTXAS_PATH` を設定して、CUDA ツールキットの ptxas を明示的に使ってください。

```shell
export CUDA_HOME=/usr/local/cuda
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
export PATH="${CUDA_HOME}/bin:$PATH"
```

## 既知の問題 { #known-issues }

- `v0.5.2`、`v0.5.3`、`v0.5.3.post1` には [zmq](https://github.com/zeromq/pyzmq/issues/2000) 起因のバグがあり、マシン構成によっては vLLM が停止することがあります。[修正](https://github.com/vllm-project/vllm/pull/6759)が含まれる最新の `vllm` にアップグレードしてください。
- 古い NCCL のメモリオーバーヘッドの問題（[バグ](https://github.com/NVIDIA/nccl/issues/1234)）に対処するため、vLLM の `>= 0.4.3, <= 0.10.1.1` では環境変数 `NCCL_CUMEM_ENABLE=0` を設定していました。vLLM に接続する外部プロセスでも、停止やクラッシュを防ぐためにこの変数を設定する必要がありました。この NCCL のバグは NCCL 2.22.3 で修正されたため、NCCL の性能最適化を活かせるよう、新しい vLLM ではこの上書きは削除されています。
- 一部の PCIe 環境（NVLink のないマシンなど）で `transport/shm.cc:590 NCCL WARN Cuda failure 217 'peer access is not supported between these two devices'` のようなエラーが出る場合、ドライバのバグが原因である可能性が高いです。詳細は[この Issue](https://github.com/NVIDIA/nccl/issues/1838) を参照してください。その場合は `NCCL_CUMEM_HOST_ENABLE=0` を設定して機能を無効にするか、ドライバを最新版に更新してみてください。
