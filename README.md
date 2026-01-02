---
title: CaptureSpace
emoji: 📱
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.10.0"
app_file: app.py
pinned: false
---

# 📱 CaptureSpace - モバイル対応物体認識アプリ

Hugging Face Spacesで動作する、iPhoneに最適化された物体認識Webアプリケーションです。

## 🌟 特徴

- **📷 モバイルカメラ対応**: iPhoneのカメラから直接撮影可能（外カメラ優先）
- **🤖 YOLOv8n使用**: 軽量で高速な物体検出モデル（CPU環境で動作）
- **🎯 リアルタイム検出**: 80種類以上の物体を認識
- **📊 JSON出力**: 検出された物体名を構造化データで取得
- **📱 レスポンシブUI**: iPhoneでの操作に最適化されたインターフェース
- **⚡ 無料で動作**: Hugging Face Spacesの無料CPU枠で利用可能

## 🚀 デプロイ済みアプリ

アプリは以下のURLで利用できます：
- **Hugging Face Spaces**: [https://huggingface.co/spaces/jin3141/CaptureSpace](https://huggingface.co/spaces/jin3141/CaptureSpace)

## 🛠️ 技術スタック

- **フレームワーク**: Gradio
- **物体検出**: Ultralytics YOLOv8n
- **画像処理**: OpenCV, Pillow
- **デプロイ**: Hugging Face Spaces

## 📦 ローカルでの実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# アプリケーションの起動
python app.py
```

ブラウザで `http://localhost:7860` にアクセスしてください。

## 🔄 GitHub Actions自動同期の設定

このリポジトリはGitHub Actionsを使用して、Hugging Face Spacesと自動同期します。

### セットアップ手順

1. **Hugging Face アクセストークンの取得**
   - [Hugging Face Settings](https://huggingface.co/settings/tokens)にアクセス
   - "New token"をクリック
   - Token type: "Write"を選択
   - トークンを生成してコピー

2. **GitHubシークレットの設定**
   - GitHubリポジトリの `Settings` > `Secrets and variables` > `Actions` に移動
   - "New repository secret"をクリック
   - Name: `HF_TOKEN`
   - Secret: コピーしたHugging Faceトークンを貼り付け
   - "Add secret"をクリック

3. **自動同期の動作**
   - `main`ブランチまたは`claude/object-detection-app-CBh50`ブランチに変更をプッシュ
   - GitHub Actionsが自動的にHugging Face Spacesに同期
   - アプリが自動的に再ビルド・デプロイされます

### 同期されるファイル

- `app.py`
- `requirements.txt`
- `.github/workflows/sync-to-huggingface.yml`

## 📝 使い方

1. アプリにアクセス
2. 「📷 カメラで撮影」ボタンをタップしてカメラを起動
3. 物体を含む写真を撮影
4. 「🔍 物体を検出」ボタンをタップ
5. バウンディングボックス付きの画像と、JSON形式の検出結果が表示されます

## 🎨 検出可能な物体

YOLOv8nモデルは、COCOデータセットの80クラスの物体を検出できます：

- 人物、動物（犬、猫、鳥など）
- 乗り物（車、バイク、飛行機など）
- 日用品（椅子、テーブル、本など）
- 電子機器（ノートPC、携帯電話、テレビなど）
- 食品（りんご、バナナ、ピザなど）

その他多数...

## 📄 ライセンス

このプロジェクトはオープンソースです。

## 🤝 貢献

Issue報告やPull Requestを歓迎します！

## 📧 お問い合わせ

質問や提案がありましたら、GitHubのIssueでお知らせください。
