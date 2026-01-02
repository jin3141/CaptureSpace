import gradio as gr
from ultralytics import YOLO
import cv2
import json
from PIL import Image
import numpy as np

# YOLOv8nモデルをロード（軽量でCPU動作に適している）
model = YOLO('yolov8n.pt')

def detect_objects(image):
    """
    画像から物体を検出し、バウンディングボックス付き画像とJSON形式の物体リストを返す

    Args:
        image: PIL Image または numpy array

    Returns:
        tuple: (バウンディングボックス付き画像, JSON文字列)
    """
    if image is None:
        return None, json.dumps({"error": "画像が提供されていません"}, ensure_ascii=False, indent=2)

    # PIL ImageをNumPy配列に変換
    if isinstance(image, Image.Image):
        image = np.array(image)

    # YOLOv8で推論実行
    results = model(image)

    # 検出された物体の名称を収集（重複なし）
    detected_objects = set()

    for result in results:
        # バウンディングボックス付きの画像を取得
        annotated_image = result.plot()

        # 検出されたクラス名を取得
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_objects.add(class_name)

    # JSONフォーマットで出力
    output_json = {
        "detected_objects": sorted(list(detected_objects))
    }
    json_string = json.dumps(output_json, ensure_ascii=False, indent=2)

    # RGB形式に変換（GradioはRGBを期待）
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    return annotated_image_rgb, json_string

# カスタムCSS（モバイル最適化）
custom_css = """
/* モバイル最適化 */
.gradio-container {
    max-width: 100% !important;
    padding: 10px !important;
}

/* ボタンを指で押しやすいサイズに */
button {
    min-height: 44px !important;
    font-size: 16px !important;
    padding: 12px 20px !important;
}

/* カメラコンポーネントのレスポンシブ調整 */
.image-container, .webcam-container {
    width: 100% !important;
    max-width: 100% !important;
}

/* テキストエリアの調整 */
textarea {
    font-size: 14px !important;
    min-height: 150px !important;
}

/* モバイルでの縦並び最適化 */
@media (max-width: 768px) {
    .gr-row {
        flex-direction: column !important;
    }

    .gr-column {
        width: 100% !important;
        max-width: 100% !important;
    }
}
"""

# Gradioインターフェースの構築
with gr.Blocks(css=custom_css, title="物体認識アプリ") as demo:
    gr.Markdown("# 📱 物体認識アプリ")
    gr.Markdown("カメラで写真を撮影して、物体を検出します。検出された物体はJSON形式で表示されます。")

    with gr.Row():
        with gr.Column():
            # カメラ入力（外カメラ優先）
            camera_input = gr.Image(
                sources=["webcam"],
                type="pil",
                label="📷 カメラで撮影",
                mirror_webcam=False  # 外カメラをデフォルトで使用
            )

            detect_btn = gr.Button("🔍 物体を検出", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            # 推論結果の画像（バウンディングボックス付き）
            output_image = gr.Image(
                label="🎯 検出結果",
                type="numpy"
            )

        with gr.Column():
            # JSON出力
            output_json = gr.Textbox(
                label="📝 検出された物体（JSON形式）",
                lines=10,
                max_lines=15
            )

    # ボタンクリック時の処理
    detect_btn.click(
        fn=detect_objects,
        inputs=[camera_input],
        outputs=[output_image, output_json],
        api_name="detect"
    )

    gr.Markdown("""
    ### 使い方
    1. 「📷 カメラで撮影」をタップしてカメラを起動
    2. 写真を撮影
    3. 「🔍 物体を検出」ボタンをタップ
    4. 検出結果とJSON形式の物体リストが表示されます

    ※ YOLOv8nモデルを使用しているため、CPU環境でも高速に動作します。
    """)

# アプリケーションの起動
if __name__ == "__main__":
    demo.queue()
    demo.launch()
