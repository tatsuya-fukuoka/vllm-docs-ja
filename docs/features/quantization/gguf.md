# GGUF { #gguf }

!!! warning
    vLLM の GGUF サポートは現時点で非常に実験的で最適化も不十分であり、他の機能と併用できない場合があります。現状はメモリ使用量を減らす手段として利用できます。問題が起きた場合は vLLM チームに報告してください。

!!! note
    GGUF のサポートはツリー外の [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin) に移行しました。GGUF のモデルをサービングする前に、GGUF プラグインがインストールされていることを確認してください。

GGUF のモデルをサービングする前に、[vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin) をインストールしてください。

```bash
uv pip install vllm-gguf-plugin
```

GGUF のモデルを vLLM で実行するには、`repo_id:quant_type` の形式で Hugging Face から直接読み込めます。たとえば [unsloth/Qwen3-0.6B-GGUF](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) から Q4_K_M の量子化モデルを読み込むには次のようにします。

```bash
# We recommend using the tokenizer from base model to avoid long-time and buggy tokenizer conversion.
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B
```

`--tensor-parallel-size 2` を追加すると、2 台の GPU でテンソル並列の推論を有効にできます。

```bash
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --tensor-parallel-size 2
```

ローカルにダウンロードした GGUF ファイルを使うこともできます。

```bash
wget https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
vllm serve ./Qwen3-0.6B-Q4_K_M.gguf --tokenizer Qwen/Qwen3-0.6B
```

!!! warning
    トークナイザーは GGUF のものではなくベースモデルのものを使うことを推奨します。GGUF からのトークナイザー変換は時間がかかり不安定で、特に語彙数の大きいモデルで顕著なためです。

GGUF は、Hugging Face がメタデータを設定ファイルへ変換できることを前提としています。Hugging Face が対象のモデルに対応していない場合は、設定を手動で用意して hf-config-path として渡せます。

```bash
# If your model is not supported by HuggingFace you can manually provide a HuggingFace compatible config path
vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
   --tokenizer Qwen/Qwen3-0.6B \
   --hf-config-path Qwen/Qwen3-0.6B
```

GGUF のモデルは、LLM のエントリポイントから直接利用することもできます。

??? code

      ```python
      from vllm import LLM, SamplingParams

      # In this script, we demonstrate how to pass input to the chat method:
      conversation = [
         {
            "role": "system",
            "content": "You are a helpful assistant",
         },
         {
            "role": "user",
            "content": "Hello",
         },
         {
            "role": "assistant",
            "content": "Hello! How can I assist you today?",
         },
         {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
         },
      ]

      # Create a sampling params object.
      sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

      # Create an LLM using repo_id:quant_type format.
      llm = LLM(
         model="unsloth/Qwen3-0.6B-GGUF:Q4_K_M",
         tokenizer="Qwen/Qwen3-0.6B",
      )
      # Generate texts from the prompts. The output is a list of RequestOutput objects
      # that contain the prompt, generated text, and other information.
      outputs = llm.chat(conversation, sampling_params)

      # Print the outputs.
      for output in outputs:
         prompt = output.prompt
         generated_text = output.outputs[0].text
         print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
      ```
