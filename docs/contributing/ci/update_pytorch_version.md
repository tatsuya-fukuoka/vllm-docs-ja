# vLLM の OSS CI/CD における PyTorch バージョンの更新 { #update-pytorch-version-on-vllm-oss-cicd }

vLLM の現在の方針は、CI/CD で常に最新の PyTorch 安定版リリースを使うことです。新しい [PyTorch 安定版リリース](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)が出たら、できるだけ早く PyTorch のバージョンを更新する PR を出すのが通例です。この作業は PyTorch のリリース間の差分が大きいため簡単ではありません。このドキュメントでは、<https://github.com/vllm-project/vllm/pull/16859> を例に、更新の一般的な手順と、起こりうる問題およびその対処法をまとめます。

## PyTorch のリリース候補（RC）でテストする { #test-pytorch-release-candidates-rcs }

正式リリース後に vLLM の PyTorch を更新するのは理想的ではありません。その時点で見つかった問題は、次のリリースを待つか、vLLM 側で場当たり的な回避策を実装するしかなくなるためです。より良い方法は、各リリースの前に PyTorch のリリース候補（RC）で vLLM をテストし、互換性を確認しておくことです。

PyTorch のリリース候補は [PyTorch のテスト用インデックス](https://download.pytorch.org/whl/test)からダウンロードできます。たとえば、`torch2.7.0+cu12.8` の RC は次のコマンドでインストールできます。

```bash
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/test/cu128
```

最終 RC がテスト可能になると、[PyTorch dev-discuss フォーラム](https://dev-discuss.pytorch.org/c/release-announcements)でコミュニティに告知されます。この告知のあと、次の 3 ステップに従ってドラフト PR を作成し、vLLM との統合テストを始められます。

1. [requirements ファイル](https://github.com/vllm-project/vllm/tree/main/requirements)を更新し、`torch`、`torchvision`、`torchaudio` の新しいリリースを指すようにします。

2. 最終リリース候補の wheel を取得するために次のオプションを使います。一般的なプラットフォームには `cpu`、`cu128`、`rocm6.2.4` などがあります。

    ```bash
    --extra-index-url https://download.pytorch.org/whl/test/<PLATFORM>
    ```

3. vLLM は `uv` を使っているため、次のインデックス戦略が適用されるようにしてください。

    - 環境変数で指定する場合:

    ```bash
    export UV_INDEX_STRATEGY=unsafe-best-match
    ```

    - CLI フラグで指定する場合:

    ```bash
    --index-strategy unsafe-best-match
    ```

PR で失敗が見つかった場合は、それを vLLM の issue として起票し、PyTorch のリリースチームを cc して対処方法の議論を始めてください。

## CUDA バージョンの更新 { #update-cuda-version }

PyTorch のリリースマトリクスには、安定版と実験的な [CUDA バージョン](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)の両方が含まれます。制約により、PyPI にアップロードされるのは最新の安定版 CUDA バージョン（たとえば torch `2.7.1+cu126`）だけです。しかし vLLM では、Blackwell をサポートするための 12.8 など、別の CUDA バージョンが必要になることがあります。そのため、そのままの `pip install torch torchvision torchaudio` コマンドを使えず、作業が複雑になります。解決策は、vLLM の Dockerfile で `--extra-index-url` を使うことです。

- 現時点で重要なインデックスは次のとおりです。

| プラットフォーム | `--extra-index-url` |
| -------- | ------------------- |
| CUDA 12.8 | [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128) |
| CPU | [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) |
| ROCm 6.2 | [https://download.pytorch.org/whl/rocm6.2.4](https://download.pytorch.org/whl/rocm6.2.4) |
| ROCm 6.3 | [https://download.pytorch.org/whl/rocm6.3](https://download.pytorch.org/whl/rocm6.3) |
| XPU | [https://download.pytorch.org/whl/xpu](https://download.pytorch.org/whl/xpu) |

- 次のファイルをステップ 1 の CUDA バージョンに合わせて更新します。これにより、リリース用の vLLM wheel が CI でテストされるようになります。
    - `.buildkite/release-pipeline.yaml`
    - `.buildkite/scripts/upload-wheels.sh`

## BuildKite CI で vLLM のビルドを手動実行する { #manually-running-vllm-builds-on-buildkiteci }

新しい PyTorch / CUDA バージョンで vLLM をビルドする場合、vLLM の sccache S3 バケットにはキャッシュされた成果物がないため、CI のビルドジョブが 5 時間を超えることがあります。さらに、vLLM の fastcheck パイプラインは読み取り専用モードで動作しキャッシュを書き込まないため、キャッシュのウォームアップには使えません。

これに対処するため、Buildkite でビルドを手動でトリガーし、次の 2 つの目的を達成します。

1. 環境変数 `RUN_ALL=1` と `NIGHTLY=1` を設定して、PyTorch RC ビルドに対してテストスイート全体を実行する
2. vLLM の sccache S3 バケットにコンパイル済み成果物を投入し、以降のビルドを高速化する

<p align="center" width="100%">
<img width="60%" alt="Buildkite new build popup" src="https://github.com/user-attachments/assets/3b07f71b-bb18-4ca3-aeaf-da0fe79d315f" />
</p>

このビルドは、[`.buildkite/scripts/trigger-ci-build.sh`](../../../.buildkite/scripts/trigger-ci-build.sh) を使ってコマンドラインからトリガーすることもできます（既定はドライラン。実際にトリガーするには `--execute` を渡します）。

## PyTorch nightly に対するテスト { #test-against-pytorch-nightly }

上記の手順では、requirements ファイルで固定した特定の PyTorch の RC / 安定版 wheel に対してテストします。代わりに最新の PyTorch **nightly** wheel に対して CI スイートをビルド・実行するには、ビルド時に環境変数 `TORCH_NIGHTLY=1` を設定します（または PR に `ready-torch-nightly` ラベルを付けます）。

`TORCH_NIGHTLY=1` の場合、CI のベースイメージが PyTorch nightly（`image_build_torch_nightly.sh`、`PYTORCH_NIGHTLY=1`、CUDA 13.0）でビルドされ、通常のイメージタグが付けられます。そのため、既存のパイプライン全体が nightly の torch 上で実行され、別途トリガーすべきパイプラインのセクションはありません。フルスイートを実行するには `RUN_ALL=1` と組み合わせてください（`ready-torch-nightly` ラベルと `trigger-ci-build.sh --torch-nightly` は、いずれもこれを自動で設定します）。定期実行の「vLLM vs PyTorch nightly」ではこの構成を使います。

コマンドラインからトリガーするには `.buildkite/scripts/trigger-ci-build.sh --torch-nightly` を使ってください。

## vLLM の各プラットフォームを更新する { #update-all-the-different-vllm-platforms }

vLLM のすべてのプラットフォームを 1 つの PR で更新しようとするより、一部のプラットフォームを分けて対応するほうが管理しやすくなります。vLLM の CI/CD ではプラットフォームごとに requirements と Dockerfile が分かれているため、更新するプラットフォームを選択できます。たとえば XPU の更新には、Intel による [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) の対応するリリースが必要です。<https://github.com/vllm-project/vllm/pull/16859> では CPU、CUDA、ROCm について vLLM を PyTorch 2.7.0 に更新し、XPU の更新は <https://github.com/vllm-project/vllm/pull/17444> で完了しました。
