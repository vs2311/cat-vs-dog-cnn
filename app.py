"""Web demo.   Usage: python app.py   (then open the printed local URL)"""
import gradio as gr

from src.predict import predict_proba


def classify(image):
    if image is None:
        return {}
    p = predict_proba(image)
    return {"dog": p, "cat": 1 - p}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a cat or dog photo"),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Classifier",
    description="CNN (MobileNetV2 transfer learning) built with TensorFlow/Keras.",
)

if __name__ == "__main__":
    demo.launch()
