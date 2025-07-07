import streamlit as st
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(page_title="🎬 Video Enhancer", layout="wide")
st.title("🎬 Manual Video Deblurring & Denoising")

video_file = st.file_uploader("📥 Upload a blurry/noisy video", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(video_file.read())
        input_path = temp_input.name

    clip = VideoFileClip(input_path)
    st.write(f"🕒 Duration: {clip.duration:.2f} sec | 🎞️ FPS: {int(clip.fps)}")

    st.header("🎛️ Enhancement Controls (Manual)")
    denoise_strength = st.slider("🧼 Denoise Strength", 0, 50, 0)
    sharpen_strength = st.slider("🔪 Sharpen Strength", 0, 5, 0)
    clahe_clip = st.slider("🌀 CLAHE Clip Limit", 0, 10, 0)

    st.markdown("## 🖼️ Preview Before Full Enhancement")
    timestamp = st.slider("📍 Preview Frame at (sec)", 0.0, clip.duration, 0.0, step=0.5)
    frame = clip.get_frame(timestamp)

    def enhance_frame(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if denoise_strength > 0:
            img = cv2.fastNlMeansDenoisingColored(img, None, denoise_strength, denoise_strength, 7, 21)
        if sharpen_strength > 0:
            kernel = np.array([[0, -1, 0],
                               [-1, 5 + sharpen_strength, -1],
                               [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
        if clahe_clip > 0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preview_enhanced = enhance_frame(frame)
    st.image(preview_enhanced, caption=f"Enhanced Preview at {timestamp} sec")

    # Full Video Enhancement
    if st.button("🚀 Enhance Full Video"):
        with st.spinner("Processing full video..."):

            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_path = temp_output.name

            cap = cv2.VideoCapture(input_path)
            fps = clip.fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enhanced = enhance_frame(rgb)
                out.write(cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
                progress.progress((i + 1) / frame_count)

            cap.release()
            out.release()

        st.success("✅ Full video enhanced successfully!")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Enhanced Video", f, file_name="enhanced_video.mp4")

else:
    st.info("📁 Upload a video to get started.")
