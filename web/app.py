import gradio as gr
import cv2
import time

def capture_video():
    cap = cv2.VideoCapture("video1.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (240, 135))
        yield frame[:, :, ::-1], frame[::-1, :, ::-1], frame[::-1, :, ::-1], frame[::-1, :, ::-1]

with gr.Blocks() as demo:
    with gr.Row():
        component = gr.Image()
        component.style(height=270, width=480)
        component1 = gr.Image()
        component1.style(height=270, width=480)
        
        component3 = gr.Image()
        component3.style(height=270, width=480)
        component4 = gr.Image()
        component4.style(height=270, width=480)
    with gr.Row():
        greet_btn = gr.Button("Greet")
        greet_btn.click(fn=capture_video, inputs=None, outputs=[component, component1, component3, component4])
# gr.Interface(capture_video, inputs=None, outputs=demo).queue().launch()
demo.queue().launch()

