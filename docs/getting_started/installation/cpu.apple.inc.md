<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM は Apple Silicon 搭載の macOS を実験的にサポートしています。現時点では、macOS 上でネイティブに実行するにはソースからビルドする必要があります。

現在、macOS 向けの CPU 実装は FP32 と FP16 のデータ型をサポートしています。

!!! tip "vLLM-Metal による GPU アクセラレーション推論"
    Metal を使った Apple Silicon 上での GPU アクセラレーション推論には、MLX を計算バックエンドとして使うコミュニティ管理のハードウェアプラグイン [vllm-metal](https://github.com/vllm-project/vllm-metal) を参照してください。

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: `macOS Sonoma` 以降
- SDK: Command Line Tools を含む `XCode 15.4` 以降
- コンパイラ: `Apple Clang >= 15.0.0`

!!! note
    macOS の CPU ビルドは、最新の GA 版 Apple Silicon ランナー上の CI でスモークテストされています。
    それ以外の macOS や Apple Clang のバージョンはベストエフォートの対応です。

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

現時点では、Apple Silicon 向けの CPU のビルド済み wheel はありません。

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

Apple Clang を含む XCode と Command Line Tools をインストールしたあと、次のコマンドを実行してソースから vLLM をビルド・インストールします。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -r requirements/cpu.txt
uv pip install -e .
```

!!! note
    macOS では `VLLM_TARGET_DEVICE` は自動的に `cpu` に設定されます。現時点でサポートされるデバイスはこれのみです。

!!! example "トラブルシューティング"
    標準の C++ ヘッダーが見つからないという次のようなエラーでビルドが失敗する場合は、
    [Command Line Tools for Xcode](https://developer.apple.com/download/all/) を削除して再インストールしてみてください。

    ```text
    [...] fatal error: 'map' file not found
            1 | #include <map>
                |          ^~~~~
        1 error generated.
        [2/8] Building CXX object CMakeFiles/_C.dir/csrc/cpu/pos_encoding.cpp.o

    [...] fatal error: 'cstddef' file not found
            10 | #include <cstddef>
                |          ^~~~~~~~~
        1 error generated.
    ```

    ---

    次のような C++11 / C++17 の互換性エラーでビルドが失敗する場合、原因はビルドシステムが古い C++ 標準を既定にしていることです。

    ```text
    [...] error: 'constexpr' is not a type
    [...] error: expected ';' before 'constexpr'
    [...] error: 'constexpr' does not name a type
    ```

    **解決策**: コンパイラが古い C++ 標準を使っている可能性があります。`cmake/cpu_extension.cmake` を編集し、`set(CMAKE_CXX_STANDARD_REQUIRED ON)` の前に `set(CMAKE_CXX_STANDARD 17)` を追加してください。

    コンパイラがサポートする C++ 標準を確認するには次のようにします。
    ```bash
    clang++ -std=c++17 -pedantic -dM -E -x c++ /dev/null | grep __cplusplus
    ```
    Apple Clang 16 では `#define __cplusplus 201703L` と表示されるはずです。

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

現時点では、Arm シリコン向けの CPU のビルド済みイメージはありません。

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
