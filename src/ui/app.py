from collections import Counter
from PIL import Image
from streamlit_image_zoom import image_zoom
from pathlib import Path
import requests
import streamlit as st

API = "http://127.0.0.1:8001"

st.title("Bone fracture detection")

TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)
LABELS = {
    "pedestrian": 0,
    "people": 1,
    "bicycle": 2,
    "car": 3,
    "van": 4,
    "truck": 5,
    "tricycle": 6,
    "awning-tricycle": 7,
    "bus": 8,
    "motor": 9,
}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Settings
with st.sidebar:
    st.title("Settings")

    selected_model = st.radio("Model:", ("Fast", "Pro"))

    classes_to_detect = st.pills(
        "Select objects to detect:", options=list(LABELS.keys()), selection_mode="multi"
    )

    sorting_order = st.radio("Sort by:", ("Upload order", "Object count"))
    if sorting_order == "Object count":
        objects_to_sort_by = st.multiselect(
            "Select objects to count",
            options=list(LABELS.keys()),
            default=list(LABELS.keys()),
        )

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
    st.session_state.uploader_key += 1  # clear the uploader
    # st.rerun() XXX
    progress_bar = st.progress(0)
    result_paths = []
    if sorting_order == "Object count":
        object_counts = []
    for idx, file in enumerate(uploaded_files):
        savepath = TMP_DIR / file.name
        with open(savepath, "wb+") as f:
            f.write(file.getvalue())
        with st.spinner("Detection in progress..."):
            response = requests.post(
                f"{API}/predict",
                json={
                    "filename": str(savepath),
                    "model": selected_model,
                    "classes": classes_to_detect,
                },
            ).json()
        progress_bar.progress((idx + 1) / len(uploaded_files))
        result_paths.append(response["output_filename"])
        if sorting_order == "Object count":
            count = Counter(response["detected"])
            object_counts.append(
                sum([count[cls] for cls in count if cls in classes_to_detect])
            )

    if sorting_order == "Object count":
        # sort by object counts descending
        result_paths = [
            path
            for _, path in sorted(
                zip(object_counts, result_paths), key=lambda pair: -pair[0]
            )
        ]
    for filename in result_paths:
        with st.expander(label=file.name, expanded=True):
            image_zoom(Image.open(filename), mode="both")
    progress_bar.empty()
