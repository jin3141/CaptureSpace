import gradio as gr
from ultralytics import YOLO
import cv2
import json
from PIL import Image
import numpy as np
import logging
import sys
from datetime import datetime
import io

# ログをキャプチャするためのStringIOハンドラー
log_stream = io.StringIO()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_stream)
    ]
)
logger = logging.getLogger(__name__)

# グローバルログリスト
app_logs = []

def add_log(message):
    """ログメッセージを追加"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    app_logs.append(log_entry)
    logger.info(message)
    return log_entry

add_log("="*60)
add_log("🚀 CaptureSpace - Object Detection App Starting")
add_log(f"📦 Gradio version: {gr.__version__}")
add_log(f"🐍 Python version: {sys.version.split()[0]}")
add_log("="*60)

# YOLOv8nモデルをロード
add_log("⏳ Loading YOLOv8n model...")
try:
    model = YOLO('yolov8n.pt')
    add_log("✅ YOLOv8n model loaded successfully")
except Exception as e:
    add_log(f"❌ Failed to load YOLO model: {e}")
    raise

def get_logs():
    """すべてのログを取得"""
    return "\n".join(app_logs[-50:])  # 最新50件のログを表示

def detect_objects(image):
    """
    画像から物体を検出し、バウンディングボックス付き画像とJSON形式の物体リストを返す
    """
    add_log("="*60)
    add_log(f"🔍 detect_objects called")
    add_log(f"📊 Image type: {type(image)}")

    try:
        if image is None:
            add_log("⚠️  No image provided")
            error_msg = json.dumps({"error": "画像が提供されていません"}, ensure_ascii=False, indent=2)
            return None, error_msg, get_logs()

        # PIL ImageをNumPy配列に変換
        if isinstance(image, Image.Image):
            add_log("🔄 Converting PIL Image to numpy array")
            image = np.array(image)

        add_log(f"📐 Image shape: {image.shape}")

        # YOLO推論実行
        add_log("🤖 Running YOLO inference...")
        results = model(image)
        add_log(f"✅ Inference completed, got {len(results)} result(s)")

        # 検出された物体の名称を収集
        detected_objects = set()
        annotated_image = None

        for i, result in enumerate(results):
            add_log(f"📋 Processing result {i+1}/{len(results)}")

            # バウンディングボックス付きの画像を取得
            annotated_image = result.plot()
            add_log(f"✅ Annotated image created, shape: {annotated_image.shape}")

            # 検出されたクラス名を取得
            if result.boxes is not None and len(result.boxes) > 0:
                num_boxes = len(result.boxes)
                add_log(f"🎯 Found {num_boxes} object(s)")

                for j, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    detected_objects.add(class_name)
                    add_log(f"  #{j+1}: {class_name} (confidence: {confidence:.2f})")
            else:
                add_log("ℹ️  No objects detected")

        # JSONフォーマットで出力
        output_json = {
            "detected_objects": sorted(list(detected_objects)),
            "count": len(detected_objects),
            "timestamp": datetime.now().isoformat()
        }
        json_string = json.dumps(output_json, ensure_ascii=False, indent=2)
        add_log(f"📝 JSON output created with {len(detected_objects)} unique objects")

        # RGB形式に変換
        if annotated_image is not None:
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            add_log(f"✅ Image converted to RGB")
        else:
            annotated_image_rgb = image

        add_log("✅ detect_objects completed successfully")
        add_log("="*60)

        return annotated_image_rgb, json_string, get_logs()

    except Exception as e:
        add_log(f"❌ ERROR in detect_objects: {e}")
        import traceback
        error_trace = traceback.format_exc()
        add_log(f"📋 Stack trace:\n{error_trace}")

        error_json = json.dumps({
            "error": str(e),
            "type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False, indent=2)
        return None, error_json, get_logs()

add_log("🔨 Creating Gradio interface...")

# Gradio Blocks with logging display
with gr.Blocks(title="CaptureSpace - 物体認識アプリ") as demo:
    gr.Markdown("# 📱 CaptureSpace - 物体認識アプリ")
    gr.Markdown("カメラで写真を撮影して、物体を検出します。**ログはリアルタイムで下部に表示されます。**")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📷 画像をアップロードまたはカメラで撮影")
            detect_btn = gr.Button("🔍 物体を検出", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            output_image = gr.Image(type="numpy", label="🎯 検出結果")
        with gr.Column():
            output_json = gr.Textbox(label="📝 検出された物体（JSON形式）", lines=10)

    with gr.Row():
        log_output = gr.Textbox(
            label="📋 アプリケーションログ（デバッグ情報）",
            lines=15,
            max_lines=20,
            value=get_logs(),
            interactive=False
        )

    gr.Markdown("""
    ### 📖 使い方
    1. 上部の画像入力エリアで画像をアップロードするか、カメラで撮影
    2. 「🔍 物体を検出」ボタンをクリック
    3. 検出結果が表示されます
    4. **ログエリアで処理の詳細を確認できます**

    ### 🐛 トラブルシューティング
    - エラーが発生した場合、ログエリアに詳細な情報が表示されます
    - 「No API Found」エラーが出る場合は、ログを確認してください
    """)

    # ボタンクリック時の処理
    detect_btn.click(
        fn=detect_objects,
        inputs=[image_input],
        outputs=[output_image, output_json, log_output]
    )

add_log("✅ Gradio interface built successfully")

if __name__ == "__main__":
    add_log("🚀 Starting application...")
    add_log("⚙️  Enabling queue...")
    demo.queue()
    add_log("✅ Queue enabled")

    add_log("🌐 Launching Gradio app...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
    add_log("✅ Application launched")
