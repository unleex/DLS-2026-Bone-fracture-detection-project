import logging
from pathlib import Path
import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.title("Bad to the backbone")

username = st.text_input("Username")
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

# Upload files
with st.container():
    col_up, col_btn = st.columns([4, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            # key=f"uploader_{st.session_state.uploader_key}",
        )
if st.button("Run!"):
    for file in uploaded_files:
        savepath = TMP_DIR / file.name
        with open(savepath, "wb+") as f:
            f.write(file.getvalue())

        r = requests.post(
            f"{API}/compute",
            json={
                "username": username,
                "filename": str(savepath),
            },
        )
        st.image(r.json()["output_filename"])


st.divider()
