# マルチモーダル対応 { #multi-modal-support }

このドキュメントでは、基本的なモデルを拡張して[マルチモーダル入力](../../features/multimodal_inputs.md)を受け付けられるようにする手順を説明します。

## 1. ベースとなる vLLM モデルの更新 { #1-update-the-base-vllm-model }

[こちらの手順](basic.md)に従って、すでに vLLM でモデルを実装済みであることを前提とします。
さらに、次のようにモデルを更新します。

- [`get_placeholder_str`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal.get_placeholder_str) を実装し、テキストプロンプト内でマルチモーダル項目を表すプレースホルダー文字列を定義します。これはモデルのチャットテンプレートと一致している必要があります。

    ??? code

        ```python
        class YourModelForImage2Seq(nn.Module):
            ...

            @classmethod
            def get_placeholder_str(cls, modality: str, i: int) -> str | None:
                if modality.startswith("image"):
                    return "<image>"

                raise ValueError("Only image modality is supported")
        ```

- `__init__` メソッド内で、モデルの言語系のコンポーネントを [`_mark_language_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal._mark_language_model) の中で、マルチモーダル系のコンポーネントを [`_mark_tower_model`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal._mark_tower_model) の中で初期化します。例:

    ```python
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()

            config = vllm_config.model_config.hf_config

            with self._mark_tower_model(vllm_config, "image"):
                self.vision_encoder = ...
                self.multi_modal_projector = ...

            with self._mark_language_model(vllm_config):
                self.language_model = init_vllm_registered_model(
                    vllm_config=vllm_config,
                    hf_config=config.text_config,
                    prefix=maybe_prefix(prefix, "language_model"),
                )
    ```

- [forward][torch.nn.Module.forward] メソッドから埋め込みの部分を取り除きます。
    - マルチモーダルの埋め込みを [`embed_multimodal`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal.embed_multimodal) へ移します。
    - テキストの埋め込みと埋め込みのマージは、[`embed_input_ids`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal.embed_input_ids) の既定の実装が自動的に処理します。ほとんどの場合、これをオーバーライドする必要はありません。

    ```diff
      def forward(
          self,
          input_ids: torch.Tensor | None,
    -     pixel_values: torch.Tensor,
          positions: torch.Tensor,
          intermediate_tensors: IntermediateTensors | None = None,
          inputs_embeds: torch.Tensor | None = None,
      ) -> torch.Tensor:
    -     if inputs_embeds is None:
    -         inputs_embeds = self.get_input_embeddings()(input_ids)
    -
    -     if pixel_values is not None:
    -         image_features = self.get_image_features(
    -             pixel_values=pixel_values,
    -         )
    -         special_image_mask = self.get_placeholder_mask(
    -             input_ids,
    -             inputs_embeds=inputs_embeds,
    -             image_features=image_features,
    -         )
    -         inputs_embeds = inputs_embeds.masked_scatter(
    -             special_image_mask,
    -             image_features,
    -         )

           hidden_states = self.language_model(
               input_ids,
               positions,
               intermediate_tensors,
               inputs_embeds=inputs_embeds,
           )
         ...
  
    +  def embed_multimodal(
    +      self,
    +      pixel_values: torch.Tensor,
    +  ) -> MultiModalEmbeddings | None:
    +      return self.get_image_features(
    +          pixel_values=pixel_values,
    +      )
    ```

    以下に [`embed_multimodal`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal.embed_multimodal) の典型的な実装パターンの雛形を示します。必要に応じて自由に調整してください。

    ```python
    def _process_image_input(self, image_input: YourModelImageInputs) -> torch.Tensor:
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings | None:
        # Validate the multimodal input keyword arguments
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        # Run multimodal inputs through encoder and projector
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
    ```

!!! important
    返される `multimodal_embeddings` は、形状 `(num_items, feature_size, hidden_size)` の **3 次元の [torch.Tensor][]**、または形状 `(feature_size, hidden_size)` の **2 次元 [torch.Tensor][] のリスト / タプル**でなければなりません。これにより、`multimodal_embeddings[i]` でリクエストの `i` 番目のマルチモーダルデータ項目（画像など）から生成された埋め込みを取得できます。

!!! note
    既定では、vLLM は入力処理で定義された [`PlaceholderRange`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/inputs/#vllm.multimodal.inputs.PlaceholderRange) の位置情報にもとづいて、
    マルチモーダルの埋め込みをテキストの埋め込みへマージします。
    このロジックは [`embed_input_ids`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal.embed_input_ids) にあります。

    埋め込みのマージ時にモデル固有の追加ロジックが必要な場合は、このメソッドをオーバーライドできます。

- 上記の手順が完了したら、モデルクラスに [`SupportsMultiModal`](https://docs.vllm.ai/en/v0.26.0/api/vllm/model_executor/models/interfaces/#vllm.model_executor.models.interfaces.SupportsMultiModal) インターフェースを追加します。

  ```diff
  + from vllm.model_executor.models.interfaces import SupportsMultiModal

  - class YourModelForImage2Seq(nn.Module):
  + class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
  ```

!!! note
    モデルクラスの名前が `*ForCausalLM` である必要はありません。
    いくつかの例は [HuggingFace Transformers のドキュメント](https://huggingface.co/docs/transformers/model_doc/auto#multimodal)を参照してください。

## 2. 処理情報の指定 { #2-specify-processing-information }

次に、[`BaseProcessingInfo`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseProcessingInfo) のサブクラスを作成し、HF の処理に関する基本情報を提供します。

### 入力項目数の上限 { #maximum-number-of-input-items }

抽象メソッド [`get_supported_mm_limits`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseProcessingInfo.get_supported_mm_limits) をオーバーライドし、モデルがサポートするモダリティごとの入力項目数の上限を返す必要があります。

たとえば、モデルが画像を任意の枚数サポートし、動画はプロンプトあたり 1 本のみサポートする場合は次のようにします。

```python
def get_supported_mm_limits(self) -> Mapping[str, int | None]:
    return {"image": None, "video": 1}
```

## 3. ダミー入力の指定 { #3-specify-dummy-inputs }

次に、[`BaseDummyInputsBuilder`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseDummyInputsBuilder) を継承して、HF の処理用のダミー入力を構築します。処理後の出力はメモリのプロファイリングにも使われます。

抽象メソッド [`get_dummy_text`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text) と [`get_dummy_mm_data`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_mm_data) をオーバーライドしてダミー入力を構築します。これらのダミー入力は、vLLM が適切な量のメモリを確保できるよう、モデルのメモリ使用量が最悪ケースになるものにすべきです。

メモリ使用量がトークン数とともに増えると仮定すると、ダミー入力は出力埋め込みの数（プレースホルダーの特徴トークン数と同じ）が最大になるように構築できます。

=== "基本的な例: LLaVA"

    HF の `LlavaForConditionalGeneration` のコードを見てみます。

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L530-L544
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0] * image_features.shape[1]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        ```

    画像 1 枚あたりのプレースホルダー特徴トークン数は `image_features.shape[1]` です。
    `image_features` は `get_image_features` メソッドの中で計算されます。

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/modeling_llava.py#L290-L300
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features
        ```

    ここから、`image_features.shape[1]` は vision tower（[`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) モデルでは `CLIPVisionModel`）の
    `image_outputs.hidden_states.shape[1]` にもとづくことが分かります。
    さらに、`image_features.shape[1]` を得るのに必要なのはシーケンス長（テンソルの 2 番目の次元）だけです。
    attention の仕組みは出力 hidden states のシーケンス長を変えないため、シーケンス長は `CLIPVisionTransformer` の
    初期 hidden states によって決まります。

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L1094-L1102
    hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
    hidden_states = self.pre_layrnorm(hidden_states)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    ```

    シーケンス長を求めるため、`CLIPVisionEmbeddings` のコードを見てみます。

    ??? code

        ```python
        # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L247-L257
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
        ```

    ここから `embeddings.shape[1] == self.num_positions` であることが分かります。ここで、

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/modeling_clip.py#L195-L196
    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.num_positions = self.num_patches + 1
    ```

    まとめると、画像 1 枚あたりのプレースホルダー特徴トークン数は次のように計算できます。

    ??? code

        ```python
        def get_num_image_tokens(
            self,
            *,
            image_width: int,
            image_height: int,
        ) -> int:
            hf_config = self.get_hf_config()
            hf_processor = self.get_hf_processor()

            image_size = hf_config.vision_config.image_size
            patch_size = hf_config.vision_config.patch_size

            num_image_tokens = (image_size // patch_size) ** 2 + 1
            if hf_processor.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            return num_image_tokens
        ```

    画像トークン数が画像の幅と高さに依存しない点に注目してください。
    マルチモーダルのプロファイリングデータの計算には、単にダミーの `image_size` を使えます。

    ??? code

        ```python
        # NOTE: In actuality, this is usually implemented as part of the
        # model's subclass of `BaseProcessingInfo`, but we show it as is
        # here for simplicity.
        def get_image_size_with_most_features(self) -> ImageSize:
            hf_config = self.get_hf_config()
            width = height = hf_config.image_size
            return ImageSize(width=width, height=height)

        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions],
        ) -> MultiModalDataDict:
            num_images = mm_counts.get("image", 0)

            target_width, target_height = \
                self.info.get_image_size_with_most_features()

            image_overrides = mm_options.get("image")

            return {
                "image": self._get_dummy_images(
                    width=target_width,
                    height=target_height,
                    num_images=num_images,
                    overrides=image_overrides,
                )
            }
        ```

    テキストについては、モデルの設定にあるマルチモーダルの画像トークンを、目的の画像枚数に合わせて単純に繰り返すだけです。

    ```python
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images
    ```

=== "入力プレースホルダーがない場合: PaliGemma"

    LLaVA とは異なり、PaliGemma の HF プロセッサは入力プロンプトに画像の
    プレースホルダートークンがあることを前提としません。プレースホルダーの特徴トークンは、
    あとから挿入されます（[プロンプトの更新](#prompt-updates)を参照）。したがって、
    ダミーのプロンプトテキストは画像枚数にかかわらず空になります。

    ```python
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""
    ```

    PaliGemma はすべての画像を `vision_config.image_size` の正方形にリサイズするため、
    画像 1 枚あたりのプレースホルダー特徴トークン数は
    `(image_size // patch_size) ** 2` に固定されます。これは PaliGemma が使う SigLIP の
    vision エンコーダによって計算されます。

    ??? code

        ```python
        # vllm/model_executor/models/siglip.py
        class SiglipEncoderInfo(VisionEncoderInfo[SiglipVisionConfig]):
            def get_num_image_tokens(
                self,
                *,
                image_width: int,
                image_height: int,
            ) -> int:
                return self.get_patch_grid_length() ** 2

            def get_patch_grid_length(self) -> int:
                image_size, patch_size = self.get_image_size(), self.get_patch_size()
                return image_size // patch_size
        ```

    画像トークン数は入力画像の寸法に依存しないため、マルチモーダルのプロファイリング
    データには、モデルが想定する入力サイズのダミー画像を単に使えばよいことになります。

    ??? code

        ```python
        def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions],
        ) -> MultiModalDataDict:
            hf_config = self.info.get_hf_config()
            vision_config = hf_config.vision_config
            max_image_size = vision_config.image_size

            num_images = mm_counts.get("image", 0)

            image_overrides = mm_options.get("image")

            return {
                "image": self._get_dummy_images(
                    width=max_image_size,
                    height=max_image_size,
                    num_images=num_images,
                    overrides=image_overrides,
                )
            }
        ```

## 4. 処理の詳細の指定 { #4-specify-processing-details }

続いて、[`BaseMultiModalProcessor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor) のサブクラスを作成し、HF の処理に関する不足情報を補います。

!!! info
    [マルチモーダルデータの処理](../../design/mm_processing.md)

### マルチモーダルのフィールド { #multi-modal-fields }

[`_get_mm_fields_config`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config) をオーバーライドし、入力のマルチモーダル項目に関連する、HF プロセッサが出力するテンソルのスキーマを返します。

=== "基本的な例: LLaVA"

    `CLIPImageProcessor` の出力は、形状
    `(num_images, num_channels, image_height, image_width)` の単純なテンソルです。


    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/clip/image_processing_clip.py#L339-L345
    images = [
        to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        for image in all_images
    ]

    data = {"pixel_values": images}
    return BatchFeature(data=data, tensor_type=return_tensors)
    ```

    そこで、[`_get_mm_fields_config`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config) を次のようにオーバーライドします。

    ```python
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )
    ```

    !!! note
        vLLM の[実際のコード](../../../vllm/model_executor/models/llava.py)は、`image_embeds` 引数で
        モデルに渡せる事前計算済みの画像埋め込みもサポートしています。

=== "後処理を伴う場合: Mistral3"

    Mistral3 の HF プロセッサが出力する `pixel_values` は、単一のテンソルへ積み上げられるよう、
    バッチ内のすべての画像を共通のサイズへパディングします。

    LLaVA と同様に [`MultiModalFieldConfig.batched`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/inputs/#vllm.multimodal.inputs.MultiModalFieldConfig.batched)
    を使うには、各画像の特徴が他の画像から独立している必要があります（これはプレフィックス
    キャッシュが正しく機能するためにも必要です）。そこで、
    [`BaseMultiModalProcessor._call_hf_processor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor)
    をオーバーライドし、各画像を元のサイズへアンパディングします。

    ??? code

        ```python
        def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
        ) -> BatchFeature:
            processed_outputs = super()._call_hf_processor(
                prompt=prompt,
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
                tok_kwargs=tok_kwargs,
            )

            pixel_values = processed_outputs.get("pixel_values")
            if pixel_values is not None:
                # Avoid padding since we need the output for each image to be
                # independent of other images for the cache to work correctly
                image_sizes = processed_outputs["image_sizes"]
                assert len(pixel_values) == len(image_sizes)

                processed_outputs["pixel_values"] = [
                    p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
                ]

            return processed_outputs
        ```

    !!! note
        `_call_hf_processor` メソッドは、処理のために `mm_kwargs` と `tok_kwargs` の両方を指定します。
        `mm_kwargs` は HuggingFace のプロセッサの初期化と呼び出しの両方に使われ、
        `tok_kwargs` は呼び出しにのみ使われます。

    `pixel_values` は画像ごとに 1 つのテンソルを持つリストになったので、
    [`_get_mm_fields_config`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_mm_fields_config) を次のようにオーバーライドできます。

    ```python
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )
    ```

    !!! note
        完全な実装は vLLM の[実際のコード](../../../vllm/model_executor/models/mistral3.py)を参照してください。

### プロンプトの更新 { #prompt-updates }

[`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) をオーバーライドし、[`PromptUpdate`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptUpdate) インスタンスのリストを返します。

各 [`PromptUpdate`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptUpdate) インスタンスは、HF プロセッサが行う更新操作（挿入、置換など）を指定します。

=== "基本的な例: LLaVA"

    HF の `LlavaProcessor` を見てみます。

    ```python
    # https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/llava/processing_llava.py#L167-L170
    prompt_strings = []
    for sample in text:
        sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
        prompt_strings.append(sample)
    ```

    入力の `image_token` を、プレースホルダー特徴トークンの数（`num_image_tokens`）と同じ回数だけ単純に繰り返しているだけです。
    これにもとづき、[`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) を次のようにオーバーライドします。

    ??? code

        ```python
        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            hf_config = self.info.get_hf_config()
            image_token_id = hf_config.image_token_index

            def get_replacement(item_idx: int):
                images = mm_items.get_items("image", ImageProcessorItems)

                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

                return [image_token_id] * num_image_tokens

            return [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_replacement,
                ),
            ]
        ```

=== "追加トークンの扱い: PaliGemma"

    PaliGemma の HF プロセッサは、プロンプト先頭の `<bos>` トークンの後ろに、画像トークンの
    連なりと、それに続いてテキストプロンプトの開始を示す 2 つ目の `<bos>` トークンを挿入します。
    まず、プレースホルダー特徴トークンごとに 1 つずつ画像トークンの連なりを構築します。

    ??? code

        ```python
        def get_insertion(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            image_tokens = [image_token_id] * num_image_tokens
            ...
        ```

    末尾の `<bos>` トークンは追加のトークンであり、vision の埋め込みを受け取っては**いけません**。
    vision の埋め込みを画像トークンにのみ割り当てるには、トークン ID を直接返すのではなく、
    [`PromptUpdateDetails`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptUpdateDetails) のインスタンスを返し、
    `embed_token_id` で埋め込み対象のトークンを指定します。

    ??? code

        ```python
        return PromptUpdateDetails.select_token_id(
            image_tokens + [bos_token_id],
            embed_token_id=image_token_id,
        )
        ```

    これらをまとめて、[`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) をオーバーライドします。
    これらのトークンは（既存のプレースホルダーを置き換えるのではなく）プロンプト先頭の `<bos>` の後ろに
    挿入されるため、接頭辞をターゲットとする [`PromptInsertion`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptInsertion)
    を使います。

    ??? code

        ```python
        def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
        ) -> Sequence[PromptUpdate]:
            hf_config = self.info.get_hf_config()
            image_token_id = hf_config.image_token_index

            tokenizer = self.info.get_tokenizer()

            bos_token_id = tokenizer.bos_token_id
            assert isinstance(bos_token_id, int)

            def get_insertion(item_idx: int):
                images = mm_items.get_items(
                    "image", (ImageEmbeddingItems, ImageProcessorItems)
                )

                if isinstance(images, ImageEmbeddingItems):
                    num_image_tokens = images.get_feature_size(item_idx)
                else:
                    image_size = images.get_image_size(item_idx)
                    num_image_tokens = self.info.get_num_image_tokens(
                        image_width=image_size.width,
                        image_height=image_size.height,
                    )

                image_tokens = [image_token_id] * num_image_tokens

                return PromptUpdateDetails.select_token_id(
                    image_tokens + [bos_token_id],
                    embed_token_id=image_token_id,
                )

            return [
                PromptInsertion(
                    modality="image",
                    target=PromptIndexTargets.prefix(
                        [bos_token_id] if tokenizer.add_bos_token else []
                    ),
                    insertion=get_insertion,
                )
            ]
        ```

## 5. プロセッサ関連クラスの登録 { #5-register-processor-related-classes }

[`BaseProcessingInfo`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseProcessingInfo)（ステップ 2）、
[`BaseDummyInputsBuilder`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseDummyInputsBuilder)（ステップ 3）、
[`BaseMultiModalProcessor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor)（ステップ 4）を定義したら、
モデルクラスを [`MULTIMODAL_REGISTRY.register_processor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/registry/#vllm.multimodal.registry.MultiModalRegistry.register_processor) でデコレートして、
マルチモーダルレジストリに登録します。

```diff
  from vllm.model_executor.models.interfaces import SupportsMultiModal
+ from vllm.multimodal import MULTIMODAL_REGISTRY

+ @MULTIMODAL_REGISTRY.register_processor(
+     YourMultiModalProcessor,
+     info=YourProcessingInfo,
+     dummy_inputs=YourDummyInputsBuilder,
+ )
  class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

## 補足 { #notes }

### 置換せずに特徴トークンを挿入する { #inserting-feature-tokens-without-replacement }

HF プロセッサの中には、元のプロンプトの何かを置き換えるのではなく、特徴トークンを直接挿入するものがあります。その場合は、[`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) の中で [`PromptReplacement`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptReplacement) の代わりに [`PromptInsertion`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.PromptInsertion) を使えます。

例:

- BLIP-2（プロンプトの先頭に挿入）: [vllm/model_executor/models/blip2.py](../../../vllm/model_executor/models/blip2.py)
- Molmo（`<|endoftext|>` トークンの後ろに挿入）: [vllm/model_executor/models/molmo.py](../../../vllm/model_executor/models/molmo.py)

### マルチモーダルデータと無関係なプロンプト更新の扱い { #handling-prompt-updates-unrelated-to-multi-modal-data }

[`_get_prompt_updates`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates) は、プロンプト更新の 1 回の適用が 1 つのマルチモーダル項目に対応することを前提としています。HF プロセッサがマルチモーダル項目の数にかかわらず追加の処理を行う場合は、[`_apply_hf_processor_tokens_only`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_tokens_only) をオーバーライドし、処理後のトークン入力が、テキスト入力に HF プロセッサを適用した結果と一致するようにしてください。これは、[vLLM の設計](../../design/mm_processing.md)ではトークン入力が HF プロセッサを迂回するためです。

例:

- Chameleon（`sep_token` を末尾に追加）: [vllm/model_executor/models/chameleon.py](../../../vllm/model_executor/models/chameleon.py)
- Molmo2（`bos_token` を先頭に追加）: [vllm/model_executor/models/molmo2.py](../../../vllm/model_executor/models/molmo2.py)
- Molmo（他所で定義されていないチャットテンプレートを適用）: [vllm/model_executor/models/molmo.py](../../../vllm/model_executor/models/molmo.py)

### カスタムの HF プロセッサ { #custom-hf-processor }

HF Hub 上で HF プロセッサのクラスを定義していないモデルもあります。その場合は、HF プロセッサと同じ呼び出しシグネチャを持つカスタムの HF プロセッサを定義し、[`_call_hf_processor`](https://docs.vllm.ai/en/v0.26.0/api/vllm/multimodal/processing/#vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor) に渡せます。

例:

- DeepSeek-VL2: [vllm/model_executor/models/deepseek_vl2.py](../../../vllm/model_executor/models/deepseek_vl2.py)
- InternVL: [vllm/model_executor/models/internvl.py](../../../vllm/model_executor/models/internvl.py)
