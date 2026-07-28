# 音声認識 API { #speech-to-text-apis }

## Transcriptions API { #transcriptions-api }

vLLM の Transcriptions API は [OpenAI の Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription) と互換性があり、
[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)からやり取りできます。

!!! note
    Transcriptions API を使うには、`pip install vllm[audio]` で音声関連の追加依存パッケージをインストールしてください。

コード例: [examples/speech_to_text/openai/openai_transcription_client.py](../../../examples/speech_to_text/openai/openai_transcription_client.py)

注意: transcriptions のエンドポイントでは、whisper などのエンコーダー・デコーダー型マルチモーダルモデルに対してビームサーチが利用できますが、エンコーダー / デコーダーのキャッシュの扱いが開発途上のため非常に非効率です。ここは現在最適化が進められており、近いうちに適切に対応される予定です。

### API で強制される上限 { #api-enforced-limits }

vLLM が受け付ける音声ファイルの最大サイズ（MB）は、環境変数
`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` で設定します。既定は 25 MB です。

### 音声ファイルのアップロード { #uploading-audio-files }

Transcriptions API は、FLAC・MP3・MP4・MPEG・MPGA・M4A・OGG・WAV・WEBM など各種形式の音声ファイルのアップロードに対応しています。

**OpenAI の Python クライアントを使う場合:**

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )

    # Upload audio file from disk
    with open("audio.mp3", "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="openai/whisper-large-v3-turbo",
            file=audio_file,
            language="en",
            response_format="verbose_json",
        )

    print(transcription.text)
    ```

**curl で multipart/form-data を使う場合:**

??? code

    ```bash
    curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
      -H "Authorization: Bearer token-abc123" \
      -F "file=@audio.mp3" \
      -F "model=openai/whisper-large-v3-turbo" \
      -F "language=en" \
      -F "response_format=verbose_json"
    ```

**サポートされるパラメータ:**

- `file`: 文字起こしする音声ファイル（必須）
- `model`: 文字起こしに使うモデル（必須）
- `language`: 言語コード（`"en"`、`"zh"` など）（任意）
- `prompt`: 文字起こしのスタイルを誘導するテキスト（任意）
- `response_format`: レスポンスの形式（`"json"`、`"text"`）（任意）
- `temperature`: 0 から 1 の間のサンプリング temperature（任意）

サンプリングパラメータや vLLM の拡張を含む完全な一覧は、[プロトコルの定義](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L2182)を参照してください。

**レスポンスの形式:**

`verbose_json` の場合のレスポンス:

??? code

    ```json
    {
      "text": "Hello, this is a transcription of the audio file.",
      "language": "en",
      "duration": 5.42,
      "segments": [
        {
          "id": 0,
          "seek": 0,
          "start": 0.0,
          "end": 2.5,
          "text": "Hello, this is a transcription",
          "tokens": [50364, 938, 428, 307, 275, 28347],
          "temperature": 0.0,
          "avg_logprob": -0.245,
          "compression_ratio": 1.235,
          "no_speech_prob": 0.012
        }
      ]
    }
    ```
現時点では、`verbose_json` のレスポンス形式は no_speech_prob に対応していません。

### 追加パラメータ { #extra-parameters }

次の[サンプリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/#inference-parameters)がサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/speech_to_text/transcription/protocol.py:transcription-sampling-params"
    ```

次の追加パラメータがサポートされています。

??? code

    ```python
    --8<-- "vllm/entrypoints/speech_to_text/transcription/protocol.py:transcription-extra-params"
    ```

## Translations API { #translations-api }

vLLM の Translation API は [OpenAI の Translations API](https://platform.openai.com/docs/api-reference/audio/createTranslation) と互換性があり、
[公式の OpenAI Python クライアント](https://github.com/openai/openai-python)からやり取りできます。
Whisper 系のモデルは、対応する 55 の非英語言語のいずれかの音声を英語に翻訳できます。
よく使われる `openai/whisper-large-v3-turbo` は翻訳に対応していない点に注意してください。

!!! note
    Translation API を使うには、`pip install vllm[audio]` で音声関連の追加依存パッケージをインストールしてください。

コード例: [examples/speech_to_text/openai/openai_translation_client.py](../../../examples/speech_to_text/openai/openai_translation_client.py)

### 追加パラメータ { #extra-parameters_1 }

次の[サンプリングパラメータ](https://docs.vllm.ai/en/v0.26.0/api/#inference-parameters)がサポートされています。

```python
--8<-- "vllm/entrypoints/speech_to_text/translation/protocol.py:translation-sampling-params"
```

次の追加パラメータがサポートされています。

```python
--8<-- "vllm/entrypoints/speech_to_text/translation/protocol.py:translation-extra-params"
```

## Realtime API { #realtime-api }

Realtime API は WebSocket ベースのストリーミング音声認識を提供し、録音しながらリアルタイムに文字起こしできます。

!!! note
    Realtime API を使うには、`uv pip install vllm[audio]` で音声関連の追加依存パッケージをインストールしてください。

### 音声の形式 { #audio-format }

音声は、サンプリングレート 16kHz・モノラルの PCM16 を base64 エンコードして送信する必要があります。

### プロトコルの概要 { #protocol-overview }

1. クライアントが `ws://host/v1/realtime` に接続する
2. サーバーが `session.created` イベントを送る
3. 必要に応じてクライアントがモデルやパラメータを含む `session.update` を送る
4. 準備ができたらクライアントが `input_audio_buffer.commit` を送る
5. クライアントが base64 の PCM16 チャンクを含む `input_audio_buffer.append` イベントを送る
6. サーバーが逐次のテキストを含む `transcription.delta` イベントを送る
7. サーバーが最終テキストと使用量を含む `transcription.done` を送る
8. 次の発話については手順 5 から繰り返す
9. 必要に応じて、クライアントが `final=True` を付けた input_audio_buffer.commit を送り、
    音声入力の終了を伝える。音声ファイルをストリーミングする場合に便利

### クライアント → サーバーのイベント { #client-server-events }

| イベント | 説明 |
| ----- | ----------- |
| `input_audio_buffer.append` | base64 エンコードした音声チャンクを送る: `{"type": "input_audio_buffer.append", "audio": "<base64>"}` |
| `input_audio_buffer.commit` | 文字起こしの処理開始または終了を指示する: `{"type": "input_audio_buffer.commit", "final": bool}` |
| `session.update` | セッションを設定する: `{"type": "session.update", "model": "model-name"}` |

### サーバー → クライアントのイベント { #server-client-events }

| イベント | 説明 |
| ----- | ----------- |
| `session.created` | セッション ID とタイムスタンプ付きで接続が確立された |
| `transcription.delta` | 逐次の文字起こしテキスト: `{"type": "transcription.delta", "delta": "text"}` |
| `transcription.done` | 使用量の統計を含む最終的な文字起こし |
| `error` | メッセージと（任意で）コードを含むエラー通知 |

#### クライアントの例 { #example-clients }

- [openai_realtime_client.py](https://github.com/vllm-project/vllm/tree/main/examples/speech_to_text/realtime/openai_realtime_client.py) - 音声ファイルをアップロードして文字起こしする
- [openai_realtime_microphone_client.py](https://github.com/vllm-project/vllm/tree/main/examples/speech_to_text/realtime/openai_realtime_microphone_client.py) - マイク入力をライブで文字起こしする Gradio のデモ
