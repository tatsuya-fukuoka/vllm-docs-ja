# 音声認識（文字起こし / 翻訳）のサポート { #speech-to-text-transcriptiontranslation-support }

このドキュメントでは、[`SupportsTranscription`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsTranscription) を実装することで、音声認識（ASR）モデルを vLLM の文字起こし / 翻訳 API に対応させる手順を説明します。詳しい指針は[対応モデル](../../models/supported_models.md#transcription)を参照してください。

## vLLM のベースモデルを更新する { #update-the-base-vllm-model }

基本的なモデルのガイドに従って、すでに vLLM でモデルを実装済みであることを前提とします。[`SupportsTranscription`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsTranscription) インターフェースでモデルを拡張し、次のクラス属性とメソッドを実装してください。

### `supported_languages` と `supports_transcription_only` { #supported_languages-and-supports_transcription_only }

対応言語と機能を宣言します。

- `supported_languages` のマッピングは初期化時に検証されます。
- モデルがテキスト生成を提供すべきでない場合（Whisper など）は `supports_transcription_only=True` を設定します。

??? code "supported_languages と supports_transcription_only"

    ```python
    from typing import ClassVar, Mapping, Literal
    import numpy as np
    import torch
    from torch import nn

    from vllm.config import ModelConfig, SpeechToTextConfig
    from vllm.inputs import PromptType
    from vllm.model_executor.models.interfaces import SupportsTranscription
    
    class YourASRModel(nn.Module, SupportsTranscription):
        # Map of ISO 639-1 language codes to language names
        supported_languages: ClassVar[Mapping[str, str]] = {
            "en": "English",
            "it": "Italian",
            # ... add more as needed
        }
        
        # If your model only supports audio-conditioned generation
        # (no text-only generation), enable this flag.
        supports_transcription_only: ClassVar[bool] = True
    ```

[`get_speech_to_text_config`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsTranscription.get_speech_to_text_config) で ASR の設定を提供します。

これは、モデルをサービングする際の API の全体的な挙動を制御するためのものです。

??? code "get_speech_to_text_config()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_speech_to_text_config(
            cls,
            model_config: ModelConfig,
            task_type: Literal["transcribe", "translate"],
        ) -> SpeechToTextConfig:
            return SpeechToTextConfig(
                sample_rate=16_000,
                max_audio_clip_s=30,
                # Set to None to disable server-side chunking if your
                # model/processor handles it already
                min_energy_split_window_size=None,
            )
    ```

各フィールドが何を制御するかは、[音声の前処理とチャンク化](#audio-preprocessing-and-chunking)を参照してください。

プロンプトの構築は [`get_generation_prompt`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsTranscription.get_generation_prompt) で実装します。サーバーは、リサンプリング済みの波形、タスクのパラメータ、リクエスト固有のオプションをまとめた [`SpeechToTextParams`](https://docs.vllm.ai/en/v0.26.0/api/vllm/config/speech_to_text/#vllm.config.speech_to_text.SpeechToTextParams) オブジェクトを構築します。モデルはこの 1 つのオブジェクトを受け取り、妥当な [`PromptType`](https://docs.vllm.ai/en/v0.26.0/api/vllm/inputs/llm/#vllm.inputs.llm.PromptType) を返します。よくあるパターンは 2 つあります。

#### 音声埋め込みを持つマルチモーダル LLM（Voxtral、Gemma3n など） { #multimodal-llm-with-audio-embeddings-eg-voxtral-gemma3n }

音声を含む `multi_modal_data` と、`prompt` の文字列または `prompt_token_ids` のいずれかを含む辞書を返します。

??? code "get_generation_prompt()"

    ```python
    from vllm.config.speech_to_text import SpeechToTextParams

    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            stt_params: SpeechToTextParams,
        ) -> PromptType:
            audio = stt_params.audio
            stt_config = stt_params.stt_config
            task_type = stt_params.task_type

            task_word = "Transcribe" if task_type == "transcribe" else "Translate"
            prompt = (
                "<start_of_turn>user\n"
                f"{task_word} this audio: <audio_soft_token>"
                "<end_of_turn>\n<start_of_turn>model\n"
            )

            return {
                "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                "prompt": prompt,
            }
    ```

    マルチモーダル入力の詳細については、[マルチモーダル入力](../../features/multimodal_inputs.md)を参照してください。

#### 音声のみのエンコーダ・デコーダ（Whisper など） { #encoderdecoder-audio-only-eg-whisper }

`encoder_prompt` と `decoder_prompt` を別々に持つ辞書を返します。

??? code "get_generation_prompt()"

    ```python
    from vllm.config.speech_to_text import SpeechToTextParams

    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_generation_prompt(
            cls,
            stt_params: SpeechToTextParams,
        ) -> PromptType:
            audio = stt_params.audio
            stt_config = stt_params.stt_config
            language = stt_params.language
            task_type = stt_params.task_type
            request_prompt = stt_params.request_prompt

            if language is None:
                raise ValueError("Language must be specified")

            prompt = {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "audio": (audio, stt_config.sample_rate),
                    },
                },
                "decoder_prompt": (
                    (f"<|prev|>{request_prompt}" if request_prompt else "")
                    + f"<|startoftranscript|><|{language}|>"
                    + f"<|{task_type}|><|notimestamps|>"
                ),
            }
            return cast(PromptType, prompt)
    ```

### `validate_language`（任意） { #validate_language-optional }

[`validate_language`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsTranscription.validate_language) による言語の検証です。

モデルが言語の指定を必要とし、既定値を設けたい場合は、このメソッドをオーバーライドしてください（Whisper を参照）。

??? code "validate_language()"

    ```python
    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            logger.warning(
                "Defaulting to language='en'. If you wish to transcribe "
                "audio in a different language, pass the `language` field "
                "in the TranscriptionRequest."
            )
            language = "en"
        return super().validate_language(language)
    ```

### `get_num_audio_tokens`（任意） { #get_num_audio_tokens-optional }

[`get_num_audio_tokens`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsTranscription.get_num_audio_tokens) による、ストリーミング時のトークン数の計上です。

ストリーミングの使用量統計の精度を上げるため、再生時間からトークン数を高速に見積もる処理を提供します。

??? code "get_num_audio_tokens()"

    ```python
    class YourASRModel(nn.Module, SupportsTranscription):
        ...

        @classmethod
        def get_num_audio_tokens(
            cls,
            audio_duration_s: float,
            stt_config: SpeechToTextConfig,
            model_config: ModelConfig,
        ) -> int | None:
            # Return None if unknown; otherwise return an estimate.
            return int(audio_duration_s * stt_config.sample_rate // 320)  # example
    ```

## 音声の前処理とチャンク化 { #audio-preprocessing-and-chunking }

API サーバーは、プロンプトを構築する前に基本的な音声の入出力と、必要に応じたチャンク化を行います。

- リサンプリング: 入力音声は `AudioResampler` によって `SpeechToTextConfig.sample_rate` にリサンプリングされます。
- チャンク化: `SpeechToTextConfig.allow_audio_chunking` が True で、再生時間が `max_audio_clip_s` を超える場合、サーバーは音声を重なりのあるチャンクに分割し、チャンクごとにプロンプトを生成します。重なりの量は `overlap_chunk_second` で制御されます。
- エネルギーを考慮した分割: `min_energy_split_window_size` が設定されている場合、サーバーはエネルギーの低い区間を探し、単語の途中で切れるのを最小限に抑えます。

関連するサーバー側のロジック:

??? code "_preprocess_speech_to_text()"

    ```python
    # vllm/entrypoints/openai/speech_to_text.py
    async def _preprocess_speech_to_text(...):
        language = self.model_cls.validate_language(request.language)
        ...
        y, sr = load_audio(bytes_, sr=self.asr_config.sample_rate)
        duration = get_audio_duration(y=y, sr=sr)
        do_split_audio = (self.asr_config.allow_audio_chunking
                        and duration > self.asr_config.max_audio_clip_s)
        chunks = [y] if not do_split_audio else self._split_audio(y, int(sr))
        prompts = []
        for chunk in chunks:
            stt_params = request.build_stt_params(
                audio=chunk,
                stt_config=self.asr_config,
                model_config=self.model_config,
                task_type=self.task_type,
            )
            prompt = self.model_cls.get_generation_prompt(stt_params)
            prompts.append(prompt)
        return prompts, duration
    ```

## タスクの自動公開 { #exposing-tasks-automatically }

モデルがこのインターフェースを実装していれば、vLLM は自動的に文字起こしのサポートを公開します。

```python
if supports_transcription(model):
    if model.supports_transcription_only:
        return ["transcription"]
    supported_tasks.append("transcription")
```

有効な場合、サーバーは文字起こしと翻訳のハンドラを初期化します。

```python
state.openai_serving_transcription = OpenAIServingTranscription(...) if "transcription" in supported_tasks else None
state.openai_serving_translation = OpenAIServingTranslation(...) if "transcription" in supported_tasks else None
```

モデルクラスがモデルレジストリ経由で利用でき、`SupportsTranscription` を実装していれば、それ以外の登録作業は不要です。

## ツリー内の例 { #examples-in-tree }

- Whisper のエンコーダ・デコーダ（音声のみ）: [vllm/model_executor/models/whisper.py](../../../vllm/model_executor/models/whisper.py)
- Voxtral のデコーダのみ（音声埋め込み + LLM）: [vllm/model_executor/models/voxtral.py](../../../vllm/model_executor/models/voxtral.py)。`mistral-common[audio]` をインストールしておいてください。
- 固定の instruction プロンプトを使う Gemma3n のデコーダのみ: [vllm/model_executor/models/gemma3n_mm.py](../../../vllm/model_executor/models/gemma3n_mm.py)
- 音声埋め込みを持つ Qwen3-Omni のマルチモーダル: [vllm/model_executor/models/qwen3_omni_moe_thinker.py](../../../vllm/model_executor/models/qwen3_omni_moe_thinker.py)

## API でのテスト { #test-with-the-api }

モデルが `SupportsTranscription` を実装したら、エンドポイントをテストできます（API は OpenAI に準拠しています）。

- 文字起こし（ASR）:

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/transcriptions
    ```

- 翻訳（別途サポートされていない限り、ソース言語 → 英語）:

    ```bash
    curl -s -X POST \
      -H "Authorization: Bearer $VLLM_API_KEY" \
      -H "Content-Type: multipart/form-data" \
      -F "file=@/path/to/audio.wav" \
      -F "model=$MODEL_ID" \
      http://localhost:8000/v1/audio/translations
    ```

その他の例は [examples/speech_to_text](../../../examples/speech_to_text) を参照してください。

!!! note
    - モデルが内部で（プロセッサやエンコーダなどで）チャンク化を行う場合は、返す `SpeechToTextConfig` で
      `min_energy_split_window_size=None` を設定し、サーバー側のチャンク化を無効にしてください。
    - `get_num_audio_tokens` を実装すると、追加の forward パスなしにストリーミングの使用量メトリクス
      （`prompt_tokens`）の精度が向上します。
    - 多言語での挙動については、`supported_languages` を実際のモデルの能力と一致させてください。
