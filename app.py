import gradio as gr
import json
from datetime import datetime

print("Starting CaptureSpace...")
print(f"Gradio version: {gr.__version__}")

def simple_test(text_input):
    """最もシンプルなテスト関数"""
    result = {
        "input": text_input,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "message": "アプリは正常に動作しています！"
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

# 最もシンプルなGradio Interface
demo = gr.Interface(
    fn=simple_test,
    inputs=gr.Textbox(label="テスト入力", placeholder="何か入力してください"),
    outputs=gr.Textbox(label="結果"),
    title="CaptureSpace - 接続テスト",
    description="このアプリが正常に動作すれば、Submitボタンを押すと結果が表示されます。"
)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
