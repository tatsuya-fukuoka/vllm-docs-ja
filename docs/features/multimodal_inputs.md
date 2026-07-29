# マルチモーダル入力 { #multimodal-inputs }

このページでは、vLLM の[マルチモーダルモデル](../models/supported_models.md#list-of-multimodal-language-models)にマルチモーダル入力を渡す方法を説明します。

!!! note
    マルチモーダル対応は現在も活発に改善が進められています。今後の変更については[この RFC](https://github.com/vllm-project/vllm/issues/4194) を参照してください。
    フィードバックや機能リクエストがあれば、[GitHub で issue を作成](https://github.com/vllm-project/vllm/issues/new/choose)してください。

!!! tip
    マルチモーダルモデルをサービングする際は、`--allowed-media-domains` を設定して vLLM がアクセスできるドメインを制限し、SSRF（サーバーサイドリクエストフォージェリ）攻撃に対して脆弱になり得る任意のエンドポイントへのアクセスを防ぐことを検討してください。この引数にはドメインのリストを指定できます。例: `--allowed-media-domains upload.wikimedia.org github.com www.bogotobogo.com`

    あわせて、ドメイン制限を回避するための HTTP リダイレクトが追跡されないよう、`VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` の設定も検討してください。

    この制限は、vLLM の Pod が内部ネットワークへ無制限にアクセスできるようなコンテナ環境で vLLM を実行する場合にとくに重要です。

## オフライン推論 { #offline-inference }

マルチモーダルデータを入力するには、[vllm.inputs.PromptType][] の次のスキーマに従います。

- `prompt`: プロンプトは HuggingFace に記載されている形式に従う必要があります。
- `multi_modal_data`: [vllm.inputs.MultiModalDataDict][] で定義されたスキーマに従う辞書です。

### 画像入力 { #image-inputs }

次の例のように、マルチモーダル辞書の `'image'` フィールドに 1 枚の画像を渡せます。

??? code

    ```python
    from vllm import LLM

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # Load the image using PIL.Image
    image = PIL.Image.open(...)

    # Single prompt inference
    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # Batch inference
    image_1 = PIL.Image.open(...)
    image_2 = PIL.Image.open(...)
    outputs = llm.generate(
        [
            {
                "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_1},
            },
            {
                "prompt": "USER: <image>\nWhat's the color of this image?\nASSISTANT:",
                "multi_modal_data": {"image": image_2},
            }
        ]
    )

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

完全な例: [examples/generate/multimodal/vision_language_offline.py](../../examples/generate/multimodal/vision_language_offline.py)

同じテキストプロンプト内で複数の画像を差し込むには、画像のリストを渡します。

??? code

    ```python
    from vllm import LLM

    llm = LLM(
        model="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,  # Required to load Phi-3.5-vision
        max_model_len=4096,  # Otherwise, it may not fit in smaller GPUs
        limit_mm_per_prompt={"image": 2},  # The maximum number to accept
    )

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "<|user|>\n<|image_1|>\n<|image_2|>\nWhat is the content of each image?<|end|>\n<|assistant|>\n"

    # Load the images using PIL.Image
    image1 = PIL.Image.open(...)
    image2 = PIL.Image.open(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [image1, image2]},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

完全な例: [examples/generate/multimodal/vision_language_multi_image_offline.py](../../examples/generate/multimodal/vision_language_multi_image_offline.py)

[LLM.chat](../models/generative_models.md#llmchat) メソッドを使う場合は、画像 URL、PIL の Image オブジェクト、事前計算済みの埋め込みなど、さまざまな形式でメッセージの content に画像を直接渡せます。

??? code

    ```python
    from vllm import LLM
    from vllm.assets.image import ImageAsset

    llm = LLM(model="llava-hf/llava-1.5-7b-hf")
    image_url = "https://picsum.photos/id/32/512/512"
    image_pil = ImageAsset('cherry_blossom').pil_image
    image_embeds = torch.load(...)

    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "image_pil",
                    "image_pil": image_pil,
                },
                {
                    "type": "image_embeds",
                    "image_embeds": image_embeds,
                },
                {
                    "type": "text",
                    "text": "What's in these images?",
                },
            ],
        },
    ]

    # Perform inference and log output.
    outputs = llm.chat(conversation)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

複数画像の入力は、動画のキャプション生成にも拡張できます。動画をサポートする [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) を使って示します。

??? code

    ```python
    from vllm import LLM

    # Specify the maximum number of frames per video to be 4. This can be changed.
    llm = LLM("Qwen/Qwen2-VL-2B-Instruct", limit_mm_per_prompt={"image": 4})

    # Create the request payload.
    video_frames = ... # load your video making sure it only has the number of frames specified earlier.
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this set of frames. Consider the frames to be a part of the same video.",
            },
        ],
    }
    for i in range(len(video_frames)):
        base64_image = encode_image(video_frames[i]) # base64 encoding.
        new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        message["content"].append(new_image)

    # Perform inference and log output.
    outputs = llm.chat([message])

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

#### RGBA の背景色のカスタマイズ { #custom-rgba-background-color }

RGBA 画像（透明度を持つ画像）を読み込むとき、vLLM はそれを RGB 形式へ変換します。既定では、透明なピクセルは白い背景に置き換えられます。この背景色は `media_io_kwargs` の `rgba_background_color` パラメータでカスタマイズできます。

??? code

    ```python
    from vllm import LLM

    # Default white background (no configuration needed)
    llm = LLM(model="llava-hf/llava-1.5-7b-hf")

    # Custom black background for dark theme
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 0]}},
    )

    # Custom brand color background (e.g., blue)
    llm = LLM(
        model="llava-hf/llava-1.5-7b-hf",
        media_io_kwargs={"image": {"rgba_background_color": [0, 0, 255]}},
    )
    ```

!!! note
    - `rgba_background_color` は RGB の値をリスト `[R, G, B]` またはタプル `(R, G, B)` で受け取ります。各値は 0〜255 です
    - この設定は透明度を持つ RGBA 画像にのみ影響します。RGB 画像は変更されません
    - 指定しない場合、後方互換性のために既定の白い背景 `(255, 255, 255)` が使われます

#### Moondream3 のプロンプトレシピ { #moondream3-prompt-recipes }

`Moondream3ForCausalLM` は、タスク別の 2 つのプロンプト形式をサポートします。

- `query`: 画像について質問する。
- `caption`: 画像のキャプションを生成する。

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

llm = LLM(
    model="moondream/moondream3-preview",
    tokenizer="moondream/starmie-v1",
    trust_remote_code=True,
    max_model_len=2048,
    limit_mm_per_prompt={"image": 1},
)

image = ImageAsset("stop_sign").pil_image


def make_query_prompt(question: str) -> str:
    return (
        "<|endoftext|><image><|md_reserved_0|>query<|md_reserved_1|>"
        f"{question}<|md_reserved_2|>"
    )


def make_caption_prompt(length: str = "normal") -> str:
    return (
        "<|endoftext|><image><|md_reserved_0|>"
        f"describe<|md_reserved_1|>{length}<|md_reserved_2|>"
    )


query_out = llm.generate(
    {
        "prompt": make_query_prompt("What is shown in this image?"),
        "multi_modal_data": {"image": image},
    },
    SamplingParams(max_tokens=64, temperature=0),
)[0].outputs[0].text

caption_out = llm.generate(
    {
        "prompt": make_caption_prompt(),
        "multi_modal_data": {"image": image},
    },
    SamplingParams(max_tokens=100, temperature=0),
)[0].outputs[0].text

print("query:", query_out)
print("caption:", caption_out)
```

!!! note
    本来の Moondream3 モデルには `detect` と `point` のスキルもあります。これらは独自の
    座標デコードを必要とし、この vLLM の実装では公開されていません。

### 動画入力 { #video-inputs }

複数画像の入力を使う代わりに、マルチモーダル辞書の `'video'` フィールドへ NumPy 配列のリストを直接渡せます。

NumPy 配列の代わりに `'torch.Tensor'` のインスタンスを渡すこともできます。Qwen2.5-VL を使った次の例を参照してください。

??? code

    ```python
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    video_path = "https://content.pexels.com/videos/free-videos.mp4"

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(max_tokens=1024)

    video_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this video."},
                {
                    "type": "video",
                    "video": video_path,
                    "total_pixels": 20480 * 28 * 28,
                    "min_pixels": 16 * 28 * 28,
                },
            ]
        },
    ]

    messages = video_messages
    processor = AutoProcessor.from_pretrained(model_path)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

    !!! note
        'process_vision_info' is only applicable to Qwen2.5-VL and similar models.

完全な例: [examples/generate/multimodal/vision_language_offline.py](../../examples/generate/multimodal/vision_language_offline.py)

### 音声入力 { #audio-inputs }

マルチモーダル辞書の `'audio'` フィールドには、タプル `(array, sampling_rate)` を渡せます。

完全な例: [examples/generate/multimodal/audio_language_offline.py](../../examples/generate/multimodal/audio_language_offline.py)

#### 文字起こし向けに長い音声を分割する { #chunking-long-audio-for-transcription }

Whisper のような音声認識モデルには、処理できる音声の長さの上限（通常 30 秒）があります。より長い音声ファイルのために、vLLM は音声を静かな箇所で賢く分割し、発話の途中で切ることを最小限に抑えるユーティリティを提供しています。

```python
from vllm import LLM, SamplingParams
from vllm.multimodal.audio import split_audio
from vllm.multimodal.media.audio import load_audio

# Load long audio file
audio, sr = load_audio("long_audio.wav", sr=16000)

# Split into chunks at low-energy (quiet) regions
chunks = split_audio(
    audio_data=audio,
    sample_rate=sr,
    max_clip_duration_s=30.0,      # Maximum chunk length in seconds
    overlap_duration_s=1.0,         # Search window for finding quiet split points
    min_energy_window_size=1600,    # Window size for energy calculation (~100ms at 16kHz)
)

# Initialize Whisper model
llm = LLM(model="openai/whisper-large-v3-turbo")
sampling_params = SamplingParams(temperature=0, max_tokens=256)

# Transcribe each chunk
transcriptions = []
for chunk in chunks:
    outputs = llm.generate({
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {"audio": (chunk, sr)},
    }, sampling_params)
    transcriptions.append(outputs[0].outputs[0].text)

# Combine results
full_transcription = " ".join(transcriptions)
```

`split_audio` 関数の特徴は次のとおりです。

- 発話の途中で切らないよう、静かな箇所で音声を分割します
- オーバーラップの範囲内で振幅の小さい領域を見つけるために RMS エネルギーを使います
- すべての音声サンプルを保持します（データの欠落なし）
- 任意のサンプリングレートに対応します

#### 音声チャンネルの自動正規化 { #automatic-audio-channel-normalization }

vLLM は、特定の音声形式を必要とするモデルのために、音声のチャンネルを自動的に正規化します。`torchaudio` のようなライブラリで音声を読み込むと、ステレオのファイルは形状 `[channels, time]` を返しますが、多くの音声モデル（とくに Whisper 系のモデル）は形状 `[time]` のモノラル音声を前提としています。

**モノラルへの自動変換に対応するモデル:**

- **Whisper** and all Whisper-based models
- **Qwen2-Audio**
- **Qwen2.5-Omni** / **Qwen3-Omni** (inherits from Qwen2.5-Omni)
- **Ultravox**

これらのモデルに対して、vLLM は自動的に次を行います。

1. 特徴抽出器を通じて、そのモデルがモノラル音声を必要とするかを検出する
2. チャンネルの平均を取って多チャンネル音声をモノラルへ変換する
3. `(channels, time)` 形式（torchaudio）と `(time, channels)` 形式（soundfile）の両方に対応する

**ステレオ音声の例:**

```python
import torchaudio
from vllm import LLM

# Load stereo audio file - returns (channels, time) shape
audio, sr = torchaudio.load("stereo_audio.wav")
print(f"Original shape: {audio.shape}")  # e.g., torch.Size([2, 16000])

# vLLM automatically converts to mono for Whisper-based models
llm = LLM(model="openai/whisper-large-v3")

outputs = llm.generate({
    "prompt": "",
    "multi_modal_data": {"audio": (audio.numpy(), sr)},
})
```

手動での変換は不要です。vLLM がモデルの要件にもとづいてチャンネルの正規化を自動的に処理します。

### 埋め込み入力 { #embedding-inputs }

あるデータ型（画像、動画、音声）に属する事前計算済みの埋め込みを言語モデルへ直接入力するには、形状 `(..., 言語モデルの hidden_size)` のテンソルをマルチモーダル辞書の対応するフィールドに渡します。
正確な形状は使用するモデルによって異なります。

この機能は `enable_mm_embeds=True` で有効にする必要があります。

!!! warning
    誤った形状の埋め込みが渡されると、vLLM のエンジンがクラッシュする可能性があります。
    このフラグは信頼できるユーザーに対してのみ有効にしてください。

#### 画像の埋め込み { #image-embeddings }

??? code

    ```python
    from vllm import LLM

    # Inference with image embeddings as input
    llm = LLM(model="llava-hf/llava-1.5-7b-hf", enable_mm_embeds=True)

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

    # For most models, `image_embeds` has shape: (num_images, image_feature_size, hidden_size)
    image_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": image_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    # Additional examples for models that require extra fields
    llm = LLM(
        "Qwen/Qwen2-VL-2B-Instruct",
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            # Shape: (total_feature_size, hidden_size)
            # total_feature_size = sum(image_feature_size for image in images)
            "image_embeds": torch.load(...),
            # Shape: (num_images, 3)
            # image_grid_thw is needed to calculate positional encoding.
            "image_grid_thw": torch.load(...),
        }
    }

    llm = LLM(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 4},
        enable_mm_embeds=True,
    )
    mm_data = {
        "image": {
            # Shape: (num_images, num_slices, hidden_size)
            # num_slices can differ for each image
            "image_embeds": [torch.load(...) for image in images],  
            # Shape: (num_images, 2)
            # image_sizes is needed to calculate details of the sliced image.
            "image_sizes": [image.size for image in images],
        }
    }
    ```

Qwen3-VL では、`image_embeds` に基本の画像埋め込みと deepstack の特徴の両方を含める必要があります。

#### 音声の埋め込み入力 { #audio-embedding-inputs }

画像の埋め込みと同様に、事前計算済みの音声埋め込みを渡せます。

??? code

    ```python
    from vllm import LLM
    import torch

    # Enable audio embeddings support
    llm = LLM(model="fixie-ai/ultravox-v0_5-llama-3_2-1b", enable_mm_embeds=True)

    # Refer to the HuggingFace repo for the correct format to use
    prompt = "USER: <audio>\nWhat is in this audio?\nASSISTANT:"

    # Load pre-computed audio embeddings, usually with shape:
    # (num_audios, audio_feature_size, hidden_size of LM)
    audio_embeds = torch.load(...)

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"audio": audio_embeds},
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)
    ```

### 入力のキャッシュ { #cached-inputs }

マルチモーダル入力を使う場合、vLLM は通常、リクエストをまたいだキャッシュを可能にするため、各メディア項目を内容にもとづいてハッシュ化します。任意で `multi_modal_uuids` を渡し、項目ごとに独自の安定した ID を与えることもできます。これにより、生データを再ハッシュすることなく、リクエストをまたいで処理結果を再利用できます。

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL example with two images
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_a = Image.open("/path/to/a.jpg")
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [img_a, img_b]},
        # Provide stable IDs for caching.
        # Requirements (matched by this example):
        #  - Include every modality present in multi_modal_data.
        #  - For lists, provide the same number of entries.
        #  - Use None to fall back to content hashing for that item.
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

UUID を使うと、該当する項目のキャッシュヒットが見込める場合に、メディアデータの送信自体を省略することもできます。省略したメディアに対応する UUID がない場合や、その UUID でキャッシュにヒットしなかった場合、リクエストは失敗する点に注意してください。

??? code

    ```python
    from vllm import LLM
    from PIL import Image

    # Qwen2.5-VL example with two images
    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = "USER: <image><image>\nDescribe the differences.\nASSISTANT:"
    img_b = Image.open("/path/to/b.jpg")

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {"image": [None, img_b]},
        # Since img_a is expected to be cached, we can skip sending the actual
        # image entirely.
        "multi_modal_uuids": {"image": ["sku-1234-a", None]},
    })

    for o in outputs:
        print(o.outputs[0].text)
    ```

!!! warning
    マルチモーダルプロセッサのキャッシュとプレフィックスキャッシュの両方が無効な場合、ユーザーが指定した `multi_modal_uuids` は無視されます。

## オンラインサービング { #online-serving }

vLLM の OpenAI 互換サーバーは、[Chat Completions API](https://platform.openai.com/docs/api-reference/chat) 経由でマルチモーダルデータを受け付けます。メディア入力では、各メディアを一意に識別するための任意の UUID もサポートしており、これはリクエストをまたいでメディアの処理結果をキャッシュするために使われます。

!!! important
    Chat Completions API を使うにはチャットテンプレートが**必要**です。
    HF 形式のモデルでは、既定のチャットテンプレートは `chat_template.json` または `tokenizer_config.json` の中で定義されています。

    既定のチャットテンプレートが利用できない場合、vLLM はまず [vllm/transformers_utils/chat_templates/registry.py](../../vllm/transformers_utils/chat_templates/registry.py) の組み込みのフォールバックを探します。
    フォールバックも利用できない場合はエラーになり、`--chat-template` 引数でチャットテンプレートを手動で指定する必要があります。

    一部のモデルについては、[examples](../../examples) に代替のチャットテンプレートを用意しています。
    たとえば VLM2Vec は [examples/pooling/embed/template/vlm2vec_phi3v.jinja](../../examples/pooling/embed/template/vlm2vec_phi3v.jinja) を使い、これは Phi-3-Vision の既定のものとは異なります。

### 画像入力 { #image-inputs_1 }

画像入力は [OpenAI Vision API](https://platform.openai.com/docs/guides/vision) に準拠してサポートされています。
Phi-3.5-Vision を使った簡単な例を示します。

まず、OpenAI 互換サーバーを起動します。

```bash
vllm serve microsoft/Phi-3.5-vision-instruct --runner generate \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt.image 2
```

次に、OpenAI クライアントを次のように使えます。

??? code

    ```python
    import os
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Single-image input inference

    # Public image URL for testing remote image processing
    image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # Create chat completion with remote image
    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    # NOTE: The prompt formatting with the image token `<image>` is not needed
                    # since the prompt will be processed automatically by the API server.
                    {
                        "type": "text",
                        "text": "What’s in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_url,  # Optional
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)

    # Local image file path (update this to point to your actual image file)
    image_file = "/path/to/image.jpg"

    # Create chat completion with local image file
    # Launch the API server/engine with the --allowed-local-media-path argument.
    if os.path.exists(image_file):
        chat_completion_from_local_image_url = client.chat.completions.create(
            model="microsoft/Phi-3.5-vision-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What’s in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{image_file}"},
                        },
                    ],
                }
            ],
        )
        result = chat_completion_from_local_image_url.choices[0].message.content
        print("Chat completion output from local image file:\n", result)
    else:
        print(f"Local image file not found at {image_file}, skipping local file test.")

    # Multi-image input inference
    image_url_duck = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"
    image_url_lion = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/lion.jpg"

    chat_response = client.chat.completions.create(
        model="microsoft/Phi-3.5-vision-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What are the animals in these images?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_duck},
                        "uuid": image_url_duck,  # Optional
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_lion},
                        "uuid": image_url_lion,  # Optional
                    },
                ],
            }
        ],
    )
    print("Chat completion output:", chat_response.choices[0].message.content)
    ```

完全な例: [examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py](../../examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)

!!! tip
    vLLM ではローカルのファイルパスからの読み込みもサポートされています。API サーバー / エンジンの起動時に `--allowed-local-media-path` で許可するローカルメディアのパスを指定し、
    API リクエストの `url` にファイルパスを渡してください。

!!! tip
    API リクエストのテキスト content に画像のプレースホルダーを置く必要はありません。画像の content ですでに表現されています。
    実際、テキストと画像の content を交互に配置することで、テキストの途中に画像を挟むこともできます。

!!! note
    HTTP URL 経由で画像を取得する際のタイムアウトは、既定で `5` 秒です。
    次の環境変数を設定して上書きできます。

    ```bash
    export VLLM_IMAGE_FETCH_TIMEOUT=<timeout>
    ```

### 動画入力 { #video-inputs_1 }

`image_url` の代わりに、`video_url` で動画ファイルを渡せます。[LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) を使った簡単な例を示します。

まず、OpenAI 互換サーバーを起動します。

```bash
vllm serve llava-hf/llava-onevision-qwen2-0.5b-ov-hf --runner generate --max-model-len 8192
```

次に、OpenAI クライアントを次のように使えます。

??? code

    ```python
    from openai import OpenAI

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    video_url = "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    ## Use video url in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this video?",
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": video_url,  # Optional
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:", result)
    ```

完全な例: [examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py](../../examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)

!!! note
    HTTP URL 経由で動画を取得する際のタイムアウトは、既定で `30` 秒です。
    次の環境変数を設定して上書きできます。

    ```bash
    export VLLM_VIDEO_FETCH_TIMEOUT=<timeout>
    ```

#### 動画デコードのバックエンド { #video-decoding-backend }

vLLM は、選択可能なデコードバックエンドを使って動画のバイト列をフレームへデコードします。3 つのバックエンドがサポートされています。

- `opencv`（既定）: OpenCV ベースのデコーダ。
- `pyav`: PyAV のデコーダ。
- `torchcodec`: TorchCodec（PyTorch ネイティブ）のデコーダ。

3 つのバックエンドはいずれも最終的に FFmpeg に支えられています。`torchcodec` では使用する FFmpeg のバージョンを選べますが、`opencv` と `pyav` はリンク時の FFmpeg のビルドに依存します。

バックエンドは、`--media-io-kwargs` で `backend` パラメータを渡して選択します。

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "torchcodec"}}'
```

**TorchCodec 固有のパラメータ:**

次のパラメータは `torchcodec` バックエンドにのみ適用されます。

- `num_ffmpeg_threads`: FFmpeg のデコードスレッド数です。`0`（既定）は FFmpeg の既定値
  `min(cpu_count + 1, 16)` に従います。スレッドの過剰割り当てを制御できます。
- `seek_mode`: デコーダのシークモードです。`"exact"`（既定）は、デコーダ作成時にファイルを
  走査することでフレーム単位の正確なサンプリングを保証します。`"approximate"` はその走査を
  省いてデコーダの作成を高速化しますが、ファイルのメタデータに依存するため（シークの精度が
  落ちる可能性があります）。

```bash
# Example: TorchCodec with approximate seek mode and 4 FFmpeg threads
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "torchcodec", "seek_mode": "approximate", "num_ffmpeg_threads": 4}}'
```

#### 動画フレームのリカバリ { #video-frame-recovery }

破損や切り詰めの可能性がある動画ファイルを処理する際の堅牢性を高めるため、vLLM は動的ウィンドウの前方スキャンによる任意のフレームリカバリをサポートしています。有効にすると、逐次読み出し中に対象フレームの読み込みが失敗した場合、（次の対象フレームより手前で）正常に取得できた次のフレームがその代わりに使われます。

動画フレームのリカバリを有効にするには、`--media-io-kwargs` で `frame_recovery` パラメータを渡します。

```bash
# Example: Enable frame recovery
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"frame_recovery": true}}'
```

**パラメータ:**

- `frame_recovery`: 前方スキャンによるリカバリを有効にする真偽値のフラグです。`true` の場合、失敗したフレームは動的ウィンドウ内（次の対象フレームまで）で利用可能な次のフレームを使って補完されます。既定値は `false` です。

**動作の仕組み:**

1. フレームを逐次読み出します
2. 対象フレームの取得に失敗した場合、それは「失敗」として記録されます
3. （次の対象フレームに達する前に）正常に取得できた次のフレームが、失敗したフレームの補完に使われます
4. この方式は、動画中間部の破損と末尾の切り詰めの両方に対応します

OpenCV バックエンドを使う場合、MP4 のような一般的な動画形式で動作します。

#### DeepStream（NVDEC）による GPU 動画デコード { #gpu-video-decoding-with-deepstream-nvdec }

既定では、vLLM は動画を CPU でデコードします。NVIDIA GPU では、代わりに DeepStream バックエンドを使ってハードウェアの動画エンジン（NVDEC）で直接デコードできます。これによりデコード処理を CPU から切り離し、動画のスループットを大きく向上させられます。

バックエンドをインストールします（Linux x86-64 のみ）。

```bash
pip install vllm[deepstream]
```

pip の wheel には DeepStream のライブラリが同梱されていますが、pip ではインストールできないいくつかのシステムパッケージに依然として依存します。Ubuntu の場合は次のとおりです。

```bash
apt-get install -y \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-libav \
  python3-gi python3-gst-1.0 libv4l-0 cuda-libraries-13-0
```

バックエンドは環境変数で選択できます。

```bash
export VLLM_VIDEO_LOADER_BACKEND=deepstream
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct
```

あるいは `--media-io-kwargs` でリクエストごとに指定します。

```bash
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "deepstream"}}'
```

**パラメータ:**

- `pool_size`: プロセス全体のデコードプールにおける GPU デコードワーカーの数です
  （`[1, 16]` にクランプされます）。未設定の場合、既定では
  `VLLM_MEDIA_LOADING_THREAD_COUNT`（既定値 `8`）になります。プールはシングルトンであるため、
  最初のリクエストの値が採用されます。

```bash
# Example: 12 decode workers
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --media-io-kwargs '{"video": {"backend": "deepstream", "pool_size": 12}}'
```

#### `media_io_kwargs` による事前抽出済みフレーム列 { #pre-extracted-frame-sequences-with-media_io_kwargs }

クライアント側で動画のフレームを抽出し、`video/jpeg`（base64 で連結した JPEG フレーム）として送信する場合、リクエストで `media_io_kwargs` を使うことで元の動画のメタデータを保持できます。これにより、クライアント側のフレーム抽出で失われてしまう時間的な情報が保たれ、より正確な動画理解が可能になります。

**サポートされるパラメータ:**

| パラメータ | 型 | 説明 |
| --------- | ---- | ----------- |
| `fps` | float | 元の動画のフレームレート |
| `frames_indices` | list[int] | 実際にサンプリングされたフレームのインデックス |
| `total_num_frames` | int | 元の動画の総フレーム数 |
| `duration` | float | 元の動画の長さ（秒） |
| `do_sample_frames` | bool | フレームのサンプリングを行うかどうか |

??? code

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    # Client-side frame extraction
    frames = extract_frames(video_path, num_frames=32)
    frames_b64 = ",".join([encode_image(f) for f in frames])
    video_url = f"data:video/jpeg;base64,{frames_b64}"

    # Pass video metadata via media_io_kwargs
    response = client.chat.completions.create(
        model="your-multimodal-model",
        messages=[{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_url}},
                {"type": "text", "text": "Describe what happens in this video."}
            ]
        }],
        extra_body={
            "media_io_kwargs": {
                "video": {
                    "fps": 30.0,
                    "frames_indices": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                       100, 110, 120, 130, 140, 150, 160, 170,
                                       180, 190, 200, 210, 220, 230, 240, 250,
                                       260, 270, 280, 290, 300, 310],
                    "total_num_frames": 900,
                    "duration": 30.0,
                }
            }
        },
    )

    print(response.choices[0].message.content)
    ```

**なぜ `media_io_kwargs` を使うのか**

クライアント側でフレームを抽出すると、サーバーは元の動画に関する重要な文脈を失います。

- **時間的な情報**: どのフレームがサンプリングされ、元のタイムライン上でどの位置にあったか
- **動画の長さ**: 元の動画がどれくらいの長さだったか
- **フレームレート**: 元の再生速度

このメタデータを渡すことで、モデルはサンプリングされたフレームの時間的な分布や、重要な瞬間が飛ばされている可能性をより良く理解できます。

#### RGBA の背景色のカスタマイズ { #custom-rgba-background-color_1 }

RGBA 画像にカスタムの背景色を使うには、`--media-io-kwargs` で `rgba_background_color` パラメータを渡します。

```bash
# Example: Black background for dark theme
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [0, 0, 0]}}'

# Example: Custom gray background
vllm serve llava-hf/llava-1.5-7b-hf \
  --media-io-kwargs '{"image": {"rgba_background_color": [128, 128, 128]}}'
```

### 音声入力 { #audio-inputs_1 }

音声入力は [OpenAI Audio API](https://platform.openai.com/docs/guides/audio?audio-generation-quickstart-example=audio-in) に準拠してサポートされています。
Ultravox-v0.5-1B を使った簡単な例を示します。

まず、OpenAI 互換サーバーを起動します。

```bash
vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b
```

次に、OpenAI クライアントを次のように使えます。

??? code

    ```python
    import base64
    import requests
    from openai import OpenAI
    from vllm.assets.audio import AudioAsset

    def encode_base64_content_from_url(content_url: str) -> str:
        """Encode a content retrieved from a remote url to base64 format."""

        with requests.get(content_url) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode('utf-8')

        return result

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Any format supported by soundfile/PyAV is supported
    audio_url = AudioAsset("winning_call").url
    audio_base64 = encode_base64_content_from_url(audio_url)

    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "wav",
                        },
                        "uuid": audio_url,  # Optional
                    },
                ],
            },
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from input audio:", result)
    ```

あるいは、画像入力における `image_url` の音声版である `audio_url` を渡すこともできます。

??? code

    ```python
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this audio?",
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                        "uuid": audio_url,  # Optional
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from audio url:", result)
    ```

完全な例: [examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py](../../examples/generate/multimodal/openai_chat_completion_client_for_multimodal.py)

!!! note
    HTTP URL 経由で音声を取得する際のタイムアウトは、既定で `10` 秒です。
    次の環境変数を設定して上書きできます。

    ```bash
    export VLLM_AUDIO_FETCH_TIMEOUT=<timeout>
    ```

### 埋め込み入力 { #embedding-inputs_1 }

あるデータ型（画像、動画、音声）に属する事前計算済みの埋め込みを言語モデルへ直接入力するには、項目ごとに形状 `(..., 言語モデルの hidden_size)` のテンソルをマルチモーダル辞書の対応するフィールドに渡します。

!!! important
    オフライン推論とは異なり、チャットテンプレートがプレースホルダートークンを正しく適用できるよう、
    各項目の埋め込みは個別に渡す必要があります。

この機能は `vllm serve` の `--enable-mm-embeds` フラグで有効にする必要があります。

!!! warning
    誤った形状の埋め込みが渡されると、vLLM のエンジンがクラッシュする可能性があります。
    このフラグは信頼できるユーザーに対してのみ有効にしてください。

#### 画像の埋め込み入力 { #image-embedding-inputs }

画像の埋め込みでは、base64 エンコードしたテンソルを `image_embeds` フィールドに渡せます。
次の例は、OpenAI サーバーへ画像の埋め込みを渡す方法を示しています。

??? code

    ```python
    from vllm.utils.serial_utils import tensor2base64

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Basic usage - this is equivalent to the LLaVA example for offline inference
    model = "llava-hf/llava-1.5-7b-hf"
    embeds = {
        "type": "image_embeds",
        "image_embeds": tensor2base64(torch.load(...)),  # Shape: (image_feature_size, hidden_size)
        "uuid": image_url,  # Optional
    }


    # Additional examples for models that require extra fields
    model = "Qwen/Qwen2-VL-2B-Instruct"
    embeds = {
        "type": "image_embeds",
        "image_embeds": {
            "image_embeds": tensor2base64(torch.load(...)),  # Shape: (image_feature_size, hidden_size)
            "image_grid_thw": tensor2base64(torch.load(...)),  # Shape: (3,)
        },
        "uuid": image_url,  # Optional
    }

    model = "openbmb/MiniCPM-V-2_6"
    embeds = {
        "type": "image_embeds",
        "image_embeds": {
            "image_embeds": tensor2base64(torch.load(...)),  # Shape: (num_slices, hidden_size)
            "image_sizes": tensor2base64(torch.load(...)),  # Shape: (2,)
        },
        "uuid": image_url,  # Optional
    }

    # Single image input
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                ],
            },
        ],
        model=model,
    )

    # Multi image input
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                    embeds,
                ],
            },
        ],
        model=model,
    )

    # Multi image input (interleaved)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": [
                    embeds,
                    {
                        "type": "text",
                        "text": "What's in this image?",
                    },
                    embeds,
                ],
            },
        ],
        model=model,
    )
    ```

### 入力のキャッシュ { #cached-inputs_1 }

オフライン推論と同様に、指定した UUID でキャッシュヒットが見込める場合は、メディアの送信を省略できます。次のようにメディアを送ることで実現できます。

??? code

    ```python
        # Image/video/audio URL:
        {
            "type": "image_url",
            "image_url": None,
            "uuid": image_uuid,
        },

        # image_embeds
        {
            "type": "image_embeds",
            "image_embeds": None,
            "uuid": image_uuid,
        },

        # input_audio:
        {
            "type": "input_audio",
            "input_audio": None,
            "uuid": audio_uuid,
        },

        # PIL Image:
        {
            "type": "image_pil",
            "image_pil": None,
            "uuid": image_uuid,
        },

    ```
