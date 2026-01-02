import gradio as gr
import json
from datetime import datetime
from PIL import Image
import numpy as np

print("Starting CaptureSpace - Step 1: Image I/O Test")
print(f"Gradio version: {gr.__version__}")

def process_image(image):
    """画像を受け取って、そのまま返す（YOLO処理なし）"""
    if image is None:
        result = {
            "error": "画像が提供されていません",
            "timestamp": datetime.now().isoformat()
        }
        return None, json.dumps(result, ensure_ascii=False, indent=2)

    # 画像をnumpy配列に変換
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image

    result = {
        "status": "success",
        "message": "画像を正常に受け取りました",
        "image_shape": str(image_array.shape),
        "image_dtype": str(image_array.dtype),
        "timestamp": datetime.now().isoformat()
    }

    # 画像をそのまま返す
    return image_array, json.dumps(result, ensure_ascii=False, indent=2)

# Gradio Interface with Image I/O
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="📷 画像をアップロードまたはカメラで撮影"),
    outputs=[
        gr.Image(type="numpy", label="🎯 処理結果"),
        gr.Textbox(label="📝 処理情報（JSON形式）", lines=10)
    ],
    title="CaptureSpace - 画像処理テスト",
    description="""
    **Step 1: 画像入出力テスト**

    画像をアップロードまたはカメラで撮影して、Submitボタンをクリックしてください。
    画像が正常に処理されることを確認します（まだ物体検出は行いません）。
    """
)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
