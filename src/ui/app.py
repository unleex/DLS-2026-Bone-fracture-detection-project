from PIL import Image
from streamlit_image_zoom import image_zoom
from pathlib import Path
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("Bad to the backbone")

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

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
            # key=f"uploader_{st.session_state.uploader_key}",
        )
# Run detection
if st.button("Run!"):
    for file in uploaded_files:
        savepath = TMP_DIR / file.name
        with open(savepath, "wb+") as f:
            f.write(file.getvalue())

        r = requests.post(
            f"{API}/compute",
            json={"filename": str(savepath), "model": selected_model},
        )
        with st.expander(label=str(savepath), expanded=True):
            image_zoom(Image.open(r.json()["output_filename"]), mode="both")
