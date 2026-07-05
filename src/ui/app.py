from PIL import Image
from streamlit_image_zoom import image_zoom
from pathlib import Path
import requests
import streamlit as st

API = "http://127.0.0.1:8001"

st.title("Aerial object detection")

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Settings
with st.sidebar:
    selected_model = st.radio("Model:", ("Fast", "Pro"))

# File upload
with st.container():
    col_up, col_btn = st.columns([4, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
# Run detection
if st.button("Run!"):
    st.session_state.uploader_key += 1
    progress_bar = st.progress(0)
    for idx, file in enumerate(uploaded_files):
        savepath = TMP_DIR / file.name
        with open(savepath, "wb+") as f:
            f.write(file.getvalue())
        with st.spinner("Detection in progress..."):
            r = requests.post(
                f"{API}/compute",
                json={"filename": str(savepath), "model": selected_model},
            )
        progress_bar.progress((idx + 1) / len(uploaded_files))
        with st.expander(label=file.name, expanded=True):
            image_zoom(Image.open(r.json()["output_filename"]), mode="both")
    progress_bar.empty()
