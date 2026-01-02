import gradio as gr
from ultralytics import YOLO
import cv2
import json
from PIL import Image
import numpy as np
import logging
import sys
from datetime import datetime

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("CaptureSpace - Object Detection App Starting")
logger.info(f"Gradio version: {gr.__version__}")
logger.info("="*60)

# YOLOv8nモデルをロード
logger.info("Loading YOLOv8n model...")
try:
    model = YOLO('yolov8n.pt')
    logger.info("✓ YOLOv8n model loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load YOLO model: {e}")
    raise

def detect_objects(image):
    """
    画像から物体を検出し、バウンディングボックス付き画像とJSON形式の物体リストを返す
    """
    logger.info("="*60)
    logger.info(f"detect_objects called at {datetime.now()}")
    logger.info(f"Image type: {type(image)}")

    try:
        if image is None:
            logger.warning("No image provided")
            error_msg = json.dumps({"error": "画像が提供されていません"}, ensure_ascii=False, indent=2)
            return None, error_msg

        # PIL ImageをNumPy配列に変換
        if isinstance(image, Image.Image):
            logger.info("Converting PIL Image to numpy array")
            image = np.array(image)

        logger.info(f"Image shape: {image.shape}")

        # YOLO推論実行
        logger.info("Running YOLO inference...")
        results = model(image)
        logger.info(f"✓ Inference completed, got {len(results)} result(s)")

        # 検出された物体の名称を収集
        detected_objects = set()
        annotated_image = None

        for i, result in enumerate(results):
            logger.info(f"Processing result {i+1}/{len(results)}")

            # バウンディングボックス付きの画像を取得
            annotated_image = result.plot()
            logger.info(f"✓ Annotated image created, shape: {annotated_image.shape}")

            # 検出されたクラス名を取得
            if result.boxes is not None and len(result.boxes) > 0:
                num_boxes = len(result.boxes)
                logger.info(f"Found {num_boxes} object(s)")

                for j, box in enumerate(result.boxes):
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    detected_objects.add(class_name)
                    logger.info(f"  Object {j+1}: {class_name} (confidence: {confidence:.2f})")
            else:
                logger.info("No objects detected")

        # JSONフォーマットで出力
        output_json = {
            "detected_objects": sorted(list(detected_objects)),
            "count": len(detected_objects),
            "timestamp": datetime.now().isoformat()
        }
        json_string = json.dumps(output_json, ensure_ascii=False, indent=2)
        logger.info(f"✓ JSON output: {json_string}")

        # RGB形式に変換
        if annotated_image is not None:
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            logger.info(f"✓ Image converted to RGB, shape: {annotated_image_rgb.shape}")
        else:
            annotated_image_rgb = image

        logger.info("✓ detect_objects completed successfully")
        logger.info("="*60)

        return annotated_image_rgb, json_string

    except Exception as e:
        logger.error(f"✗ Error in detect_objects: {e}", exc_info=True)
        error_json = json.dumps({
            "error": str(e),
            "type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }, ensure_ascii=False, indent=2)
        return None, error_json

logger.info("Creating Gradio interface...")

# Gradio Interface（Blocksではなく）を使用
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="📷 画像をアップロードまたはカメラで撮影"),
    outputs=[
        gr.Image(type="numpy", label="🎯 検出結果"),
        gr.Textbox(label="📝 検出された物体（JSON形式）", lines=10)
    ],
    title="📱 CaptureSpace - 物体認識アプリ",
    description="""
    ## 使い方
    1. 画像をアップロードするか、カメラで写真を撮影してください
    2. 「Submit」ボタンをクリック
    3. 検出結果とJSON形式の物体リストが表示されます

    ※ YOLOv8nモデルを使用（CPU環境で高速動作）
    """,
    examples=None,
    allow_flagging="never",
    analytics_enabled=False
)

logger.info("✓ Gradio Interface created successfully")

if __name__ == "__main__":
    logger.info("Starting application...")
    logger.info("Enabling queue...")
    demo.queue()
    logger.info("✓ Queue enabled")

    logger.info("Launching Gradio app...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
    logger.info("✓ Application launched")
