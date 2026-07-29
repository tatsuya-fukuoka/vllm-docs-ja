# コミッター { #committers }

このドキュメントでは、vLLM プロジェクトの現在のコミッターと、各自が担当する主要領域を一覧にしています。コミッターは vLLM リポジトリへの書き込み権限を持ち、PR のレビューとマージを担当します。ファイル単位の具体的なオーナーシップとレビュアーについては、[CODEOWNERS](https://github.com/vllm-project/vllm/blob/main/.github/CODEOWNERS) ファイルも参照してください。このドキュメントと CODEOWNERS ファイルはどちらも生きたドキュメントであり、互いを補完しています。

## 現役のコミッター { #active-committers }

各コミッターの vLLM における役割を数語でまとめています。一般に、vLLM のコミッターは幅広い領域をカバーし、メンテナンスの過程で互いに助け合っています。コンポーネントごとの正確な担当については、後述の「領域オーナー」のセクションを参照してください。GitHub ハンドルのアルファベット順に並べています。

- [@22quinn](https://github.com/22quinn): RL API
- [@aarnphm](https://github.com/aarnphm): 構造化出力
- [@alexm-redhat](https://github.com/alexm-redhat): 性能
- [@ApostaC](https://github.com/ApostaC): コネクタ、オフロード
- [@bbrowning](https://github.com/bbrowning): ツール利用と推論パーサー
- [@benchislett](https://github.com/benchislett): エンジンコア、投機的デコーディング
- [@bigPYJ1151](https://github.com/bigPYJ1151): Intel CPU/XPU の統合
- [@BugenZhao](https://github.com/BugenZhao): Rust フロントエンド
- [@chaunceyjiang](https://github.com/chaunceyjiang): ツール利用と推論パーサー
- [@DarkLight1337](https://github.com/DarkLight1337): マルチモーダル、API サーバー
- [@esmeetu](https://github.com/esmeetu): 開発者向けマーケティング、コミュニティ
- [@dllehr-amd](https://github.com/dllehr-amd): AMD の統合
- [@heheda12345](https://github.com/heheda12345): ハイブリッドメモリアロケータ
- [@hmellor](https://github.com/hmellor): Hugging Face の統合、ドキュメント
- [@houseroad](https://github.com/houseroad): エンジンコア、Llama モデル
- [@Isotr0py](https://github.com/Isotr0py): マルチモーダル、新規モデル対応
- [@jeejeelee](https://github.com/jeejeelee): LoRA、新規モデル対応
- [@jikunshang](https://github.com/jikunshang): Intel CPU/XPU の統合
- [@khluu](https://github.com/khluu): CI 基盤
- [@KuntaiDu](https://github.com/KuntaiDu): KV コネクタ
- [@LucasWilkinson](https://github.com/LucasWilkinson): カーネルと性能
- [@luccafong](https://github.com/luccafong): Llama モデル、投機的デコーディング、分散
- [@markmc](https://github.com/markmc): オブザーバビリティ
- [@MatthewBonanni](https://github.com/MatthewBonanni): カーネルと性能
- [@mgoin](https://github.com/mgoin): 量子化と性能
- [@NickLucche](https://github.com/NickLucche): KV コネクタ
- [@njhill](https://github.com/njhill): 分散、API サーバー、エンジンコア
- [@noooop](https://github.com/noooop): プーリングモデル
- [@patrickvonplaten](https://github.com/patrickvonplaten): Mistral モデル、新規モデル対応
- [@pavanimajety](https://github.com/pavanimajety): NVIDIA GPU の統合
- [@ProExpertProg](https://github.com/ProExpertProg): コンパイル、起動時の UX
- [@robertgshaw2-redhat](https://github.com/robertgshaw2-redhat): コア、分散、分離配置
- [@ruisearch42](https://github.com/ruisearch42): パイプライン並列、Ray 対応
- [@russellb](https://github.com/russellb): 構造化出力、エンジンコア、セキュリティ
- [@sfeng33](https://github.com/sfeng33): ツール利用と推論パーサー
- [@sighingnow](https://github.com/sighingnow): Qwen モデル、新規モデル対応
- [@simon-mo](https://github.com/simon-mo): プロジェクトリード、API エントリポイント、コミュニティ
- [@tdoublep](https://github.com/tdoublep): 状態空間モデル
- [@tjtanaa](https://github.com/tjtanaa): AMD GPU の統合
- [@tlrmchlsmth](https://github.com/tlrmchlsmth): カーネルと性能、分散、分離配置
- [@WoosukKwon](https://github.com/WoosukKwon): プロジェクトリード、エンジンコア
- [@yaochengji](https://github.com/yaochengji): TPU の統合
- [@yeqcharlotte](https://github.com/yeqcharlotte): ベンチマーク、Llama モデル
- [@yewentao256](https://github.com/yewentao256): カーネルと性能
- [@Yikun](https://github.com/Yikun): 差し替え可能なハードウェアインターフェース
- [@youkaichao](https://github.com/youkaichao): プロジェクトリード、分散、コンパイル、コミュニティ
- [@ywang96](https://github.com/ywang96): マルチモーダル、ベンチマーク
- [@zhuohan123](https://github.com/zhuohan123): プロジェクトリード、RL 統合、数値計算
- [@zou3519](https://github.com/zou3519): コンパイル
- [@BoyuanFeng](https://github.com/BoyuanFeng): コンパイル、CUDA graph
- [@xuechendi](https://github.com/xuechendi): Intel CPU/XPU の統合、KV コネクタ

### 名誉コミッター { #emeritus-committers }

過去に vLLM へ大きく貢献してくれた（ありがとうございます）ものの、現在は活動していないコミッターです。

- [@andoorve](https://github.com/andoorve): パイプライン並列
- [@cadedaniel](https://github.com/cadedaniel): 投機的デコーディング
- [@comaniac](https://github.com/comaniac): KV キャッシュ管理、パイプライン並列
- [@LiuXiaoxuanPKU](https://github.com/LiuXiaoxuanPKU): 投機的デコーディング
- [@pcmoritz](https://github.com/pcmoritz): MoE
- [@rkooo567](https://github.com/rkooo567): チャンク化プレフィル
- [@sroy745](https://github.com/sroy745): 投機的デコーディング
- [@Yard1](https://github.com/Yard1): カーネルと性能
- [@zhisbug](https://github.com/zhisbug): Arctic モデル、分散

## 領域オーナー { #area-owners }

このセクションでは、現役コミッターを vLLM のコンポーネント別に整理し、各領域のオーナーを示します。その領域に触れる PR がある場合は、遠慮なく領域オーナーにレビューを依頼してください。

### エンジンコア { #engine-core }

- スケジューラ: 次のバッチにリクエストを割り当てる vLLM エンジンの中核ループ
    - @WoosukKwon, @robertgshaw2-redhat, @njhill, @heheda12345
- KV キャッシュマネージャー: KV キャッシュの論理ブロックデータを管理する、スケジューラ内のメモリ管理層
    - @heheda12345, @WoosukKwon
- AsyncLLM: エンジンコアをホストし、エントリポイントから利用できるようにする ZMQ ベースのプロトコル
    - @robertgshaw2-redhat, @njhill, @russellb
- ModelRunner / Executor / Worker: モデル実装をラップするエンジン側の抽象
    - @WoosukKwon, @tlrmchlsmth, @heheda12345, @LucasWilkinson, @ProExpertProg, @MatthewBonanni
- KV コネクタ: KV キャッシュのオフロードと転送のためのコネクタのインターフェースと実装
    - @robertgshaw2-redhat, @njhill, @KuntaiDu, @NickLucche, @ApostaC
- 分散 / 並列 / プロセス管理: 各ワーカーを管理し、適切な DP/TP/PP/EP のランクに割り当てるプロセスランチャー
    - @youkaichao, @njhill, @WoosukKwon, @ruisearch42
- 集団通信: NCCL などの通信ライブラリ / カーネルの利用
    - @tlrmchlsmth, @youkaichao
- マルチモーダルのエンジンとメモリ管理: 画像・音声・動画の入力に関わるスケジューリングとメモリ管理
    - @ywang96, @DarkLight1337

### モデル実装 { #model-implementations }

- モデルインターフェース: 各種モデルの `nn.Module` インターフェースと実装
    - @zhuohan123, @mgoin, @simon-mo, @houseroad, @ywang96 (multimodality), @jeejeelee (lora)
- Logits Processor / Sampler: 提供されるサンプラークラスと差し替え可能な logits processor
    - @njhill, @houseroad, @22quinn
- カスタム層: rotary embedding や RMS norm など、vLLM のユーティリティ層
    - @ProExpertProg
- Attention: paged attention のための Attention インターフェース
    - @WoosukKwon, @LucasWilkinson, @heheda12345, @MatthewBonanni
- FusedMoE: FusedMoE カーネル、モジュラーカーネルのフレームワーク、EPLB
    - @tlrmchlsmth
- 量子化: 各種の量子化設定、重みの読み込み、カーネル
    - @mgoin, @Isotr0py, @yewentao256
- カスタムの量子化 GEMM カーネル（cutlass_scaled_mm、marlin、machete）
    - @tlrmchlsmth, @LucasWilkinson
- マルチモーダル入力の処理: 画像 / 動画 / 音声のデータを読み込み、特徴テンソルに変換するコンポーネント
    - @DarkLight1337, @ywang96, @Isotr0py
- torch compile: vLLM における torch.compile の統合、カスタムパスと変換
    - @ProExpertProg, @zou3519, @youkaichao, @BoyuanFeng
- 状態空間モデル: vLLM における状態空間モデルの実装
    - @tdoublep, @tlrmchlsmth
- 推論およびツール呼び出しのパーサー
    - @chaunceyjiang, @aarnphm, @sfeng33, @bbrowning

### エントリポイント { #entrypoints }

- LLM クラス: オフライン推論のための LLM クラス
    - @DarkLight1337
- API サーバー: OpenAI 互換の API サーバー
    - @DarkLight1337, @njhill, @aarnphm, @simon-mo, @heheda12345 (Responses API)
- Rust フロントエンド: Rust による実験的な API サーバー
    - @BugenZhao, @njhill
- バッチランナー: OpenAI 互換のバッチランナー
    - @simon-mo

### 機能 { #features }

- 投機的デコーディング: n-gram、EAGLE、MTP に関わるモデル定義、Attention、サンプラー、スケジューラ
    - @WoosukKwon, @benchislett, @luccafong, @MatthewBonanni
- 構造化出力: 構造化出力の実装
    - @russellb, @aarnphm
- RL: collective rpc やスリープモードなど、RL 関連の機能
    - @youkaichao, @zhuohan123, @22quinn
- LoRA: @jeejeelee
- オブザーバビリティ: メトリクスとロギング
    - @markmc, @robertgshaw2-redhat, @simon-mo

### コードベース { #code-base }

- 設定: 設定の登録と解析
    - @hmellor
- ドキュメント: @hmellor, @DarkLight1337, @simon-mo
- ベンチマーク: @ywang96, @simon-mo
- CI・ビルド・リリースプロセス: @khluu, @njhill, @simon-mo
- セキュリティ: @russellb

### 外部カーネルの統合 { #external-kernels-integration }

- FlashAttention: @LucasWilkinson, @MatthewBonanni
- FlashInfer: @LucasWilkinson, @mgoin, @WoosukKwon, @MatthewBonanni
- Blackwell 向けカーネル: @mgoin, @yewentao256
- DeepEP/DeepGEMM: @mgoin, @yewentao256

### 各種統合 { #integrations }

- Hugging Face: @hmellor, @Isotr0py
- Ray: @ruisearch42
- NIXL: @robertgshaw2-redhat, @NickLucche

### モデルベンダーとの協業 { #collaboration-with-model-vendors }

- gpt-oss: @heheda12345, @simon-mo, @zhuohan123
- Llama: @luccafong
- Qwen: @sighingnow
- Mistral: @patrickvonplaten

### ハードウェア { #hardware }

- プラグインインターフェース: @youkaichao, @Yikun
- NVIDIA GPU: @pavanimajety
- AMD GPU: @gshtras, @tjtanaa
- Intel CPU/GPU: @jikunshang, @bigPYJ1151, @xuechendi
- Google TPU: @yaochengji

### エコシステムのプロジェクト { #ecosystem-projects }

- Ascend NPU: [@wangxiyuan](https://github.com/wangxiyuan)、[詳細はこちら](https://vllm-ascend.readthedocs.io/en/latest/community/contributors.html#maintainers)
- Intel Gaudi HPU: [@xuechendi](https://github.com/xuechendi)、[@kzawora-intel](https://github.com/kzawora-intel)
- Semantic Router: [@xunzhuo](https://github.com/xunzhuo)、[@rootfs](https://github.com/rootfs)、[詳細はこちら](https://vllm-semantic-router.com/community/team)
