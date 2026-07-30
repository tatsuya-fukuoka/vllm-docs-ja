# CI の失敗 { #ci-failures }

自分の PR で CI ジョブが失敗したものの、その失敗が自分の PR に起因するとは思えない場合、どうすればよいでしょうか。

- 現在の CI テスト失敗のダッシュボードを確認してください。  
  👉 [CI Failures Dashboard](https://github.com/orgs/vllm-project/projects/20)

- その失敗が**すでに掲載されている**場合、あなたの PR とは無関係である可能性が高いです。
  修正への協力はいつでも歓迎します。
    - 同じ失敗の別の発生例へのリンクをコメントとして残してください。
    - 👍 のリアクションを付けて、影響を受けている人数を示してください。

- その失敗が**掲載されていない**場合は、**issue を作成**してください。

## CI テスト失敗の issue を作成する { #filing-a-ci-test-failure-issue }

- **バグ報告を作成する:**  
    👉 [New CI Failure Report](https://github.com/vllm-project/vllm/issues/new?template=450-ci-failure.yml)

- **タイトルは次の形式にしてください:**

    ```text
    [CI Failure]: failing-test-job - regex/matching/failing:test
    ```

- **environment のフィールドには次のように書きます:**

    ```text
    Still failing on main as of commit abcdef123
    ```

- **本文には失敗しているテストを含めます:**

    ```text
    FAILED failing/test.py:failing_test1 - Failure description
    FAILED failing/test.py:failing_test2 - Failure description
    https://github.com/orgs/vllm-project/projects/20
    https://github.com/vllm-project/vllm/issues/new?template=400-bug-report.yml
    FAILED failing/test.py:failing_test3 - Failure description
    ```

- **ログを添付します**（折りたたみセクションの例）:
    <details>
    <summary>Logs:</summary>

    ```text
    ERROR 05-20 03:26:38 [dump_input.py:68] Dumping input data
    --- Logging error ---  
    Traceback (most recent call last):  
      File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 203, in execute_model  
        return self.model_executor.execute_model(scheduler_output)
    ...
    FAILED failing/test.py:failing_test1 - Failure description
    FAILED failing/test.py:failing_test2 - Failure description
    FAILED failing/test.py:failing_test3 - Failure description
    ```

    </details>

## ログの扱い { #logs-wrangling }

ログは公開されており、Buildkite へのログインは不要です。
[.buildkite/scripts/ci-fetch-log.sh](../../../.buildkite/scripts/ci-fetch-log.sh)
は各ログを `ci-<build>-<job-name>.log` として保存し、タイムスタンプと ANSI コードを取り除きます。

```bash
# All failed jobs in a PR's latest build (current branch's PR if omitted):
.buildkite/scripts/ci-fetch-log.sh --pr <PR>

# All failed jobs in a build (--soft also includes soft-failed jobs;
# --all fetches every finished job):
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>"

# One job — `gh pr checks` URLs (#<job_uuid>) and web UI URLs (?sid=) both
# work; pass "-" as a second argument to stream to stdout:
.buildkite/scripts/ci-fetch-log.sh "https://buildkite.com/vllm/ci/builds/<N>#<job_uuid>"
```

すでにダウンロード済みのログを整形するには次のようにします。

[.buildkite/scripts/ci-clean-log.sh](../../../.buildkite/scripts/ci-clean-log.sh)

```bash
./ci-clean-log.sh ci.log
```

すばやくコピー＆ペーストするには [wl-clipboard](https://github.com/bugaevc/wl-clipboard) が便利です。

```bash
tail -525 ci_build.log | wl-copy
```

## CI テスト失敗の調査 { #investigating-a-ci-test-failure }

1. 👉 [Buildkite main branch](https://buildkite.com/vllm/ci/builds?branch=main) を開きます
2. 二分探索で、問題が最初に現れたビルドを特定します。  
3. 調査結果を GitHub の issue に追記します。  
4. 有力な候補の PR が見つかったら、issue でそれに言及し、コントリビューターにメンションしてください。

## 失敗の再現 { #reproducing-a-failure }

CI のテスト失敗は不安定（flaky）なことがあります。bash のループで繰り返し実行してください。

[.buildkite/scripts/rerun-test.sh](../../../.buildkite/scripts/rerun-test.sh)

```bash
./rerun-test.sh tests/v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]
```

## PR の提出 { #submitting-a-pr }

CI の失敗を修正する PR を提出する場合は次のようにしてください。

- PR を issue に紐づける:
  PR の説明に `Closes #12345` を追加します。
- `ci-failure` ラベルを付ける:
  [CI Failures の GitHub Project](https://github.com/orgs/vllm-project/projects/20) で追跡しやすくなります。

## その他のリソース { #other-resources }

- 🔍 [`main` におけるテストの信頼性](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&order=ASC&sort_by=reliability)
- 🧪 [最新の Buildkite CI 実行](https://buildkite.com/vllm/ci/builds?branch=main)

## 日次のトリアージ { #daily-triage }

[Buildkite analytics（2 日間ビュー）](https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&period=2days) を使って次のことを行います。

- **`main` 上での**最近のテスト失敗を特定する。
- PR 上の正当なテスト失敗を除外する。
- （任意）信頼性 0% のテストを無視する。

その結果を [CI Failures Dashboard](https://github.com/orgs/vllm-project/projects/20) と比較してください。
