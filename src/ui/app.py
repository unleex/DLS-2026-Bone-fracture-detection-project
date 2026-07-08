import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import json
from pathlib import Path
from PIL import Image
import requests
import streamlit as st
from streamlit_image_zoom import image_zoom
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

API = "http://0.0.0.0:8000"

st.title("Aerial image analysis AI")

# TODO: clear tmp files
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

if "results" not in st.session_state:
    st.session_state.results = []


# File upload
def clear_uploader():
    st.session_state.uploader_key += 1


with st.container():
    upload_col, clear_col = st.columns([6, 1])
    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
    with clear_col:
        # Align the button with the uploader
        st.text("")
        st.text("")
        st.button("Clear files", on_click=clear_uploader)


# Settings
with st.sidebar:
    st.title("⚙️ Settings")

    selected_model = st.radio("Model:", ("Fast", "Pro"))

    selected_classes = st.pills(
        "Select objects to detect:",
        options=list(LABELS.keys()),
        selection_mode="multi",
        default=list(LABELS.keys()),
    )

    sorting_order = st.radio("Sort by:", ("Upload order", "Object count"))
    if sorting_order == "Object count":
        objects_to_sort_by = st.multiselect(
            "Select objects to count",
            options=selected_classes,
            default=selected_classes,
        )


# Run detection
if st.button("Run!"):
    progress_bar = st.progress(0)
    st.session_state.results = []

    for idx, file in enumerate(uploaded_files):
        savepath = TMP_DIR / file.name
        with open(savepath, "wb+") as f:
            f.write(file.getvalue())

        with st.spinner(f"Processing {file.name}..."):
            r = requests.post(
                f"{API}/predict",
                json={
                    "filename": str(savepath),
                    "model": selected_model,
                },
            )
        if not r.ok:
            print(r)
            st.error("❌ Internal error occured, try again later")
        # Parse detection results
        response = r.json()
        raw_results = response["results"]
        detections = (
            json.loads(raw_results) if isinstance(raw_results, str) else raw_results
        )

        st.session_state.results.append(
            {
                "original_name": file.name,
                "output_filename": response["output_filename"],
                "detections": detections,
            }
        )
        progress_bar.progress((idx + 1) / len(uploaded_files))

    progress_bar.empty()


# Post-processing and rendering. Triggers dynamically on sidebar change
if st.session_state.results:
    results = []
    detection_counts: list[Counter] = []

    # Select classes based on the pills
    for res in st.session_state.results:
        filtered = [d for d in res["detections"] if d["name"] in selected_classes]
        results.append(
            {
                "original_name": res["original_name"],
                "output_filename": res["output_filename"],
                "detections": filtered,
            }
        )
        detection_counts.append(Counter([d["name"] for d in filtered]))

    # Sort if specified
    if sorting_order == "Object count":

        def result_sorter(result):
            return sum(
                1 for d in result["detections"] if d["name"] in objects_to_sort_by
            )

        # Arrange detection_counts in the same order results are sorted in
        to_sort = list(zip(results, detection_counts, strict=True))
        to_sort.sort(key=lambda item: result_sorter(item[0]), reverse=True)
        results, detection_counts = zip(*to_sort)

    # Gather and display statistics
    mean_counts = pd.DataFrame(columns=["count"])
    for counts in detection_counts:
        new_counts = pd.DataFrame({"count": counts.values()}, index=counts.keys())
        concatenated = pd.concat([mean_counts, new_counts])
        mean_counts = (
            pd.DataFrame(detection_counts).fillna(0).mean().to_frame(name="count")
        )
    total = mean_counts["count"].sum()
    ax = mean_counts.plot.pie(
        "count",
        autopct=lambda p: f"{int(round(p * total / 100))}",
        title="Average number of objects found per image",
        radius=0.7,
    )
    ax.legend(loc=2, prop={"size": 6})
    with st.expander(label="🧮 Statistics", expanded=False):
        st.pyplot(ax.figure)

    # Display annotated images
    for res, counts in zip(results, detection_counts, strict=True):
        with st.expander(label="▶️ " + res["original_name"], expanded=True):
            # Load the original, unannotated image
            original_img_path = TMP_DIR / res["original_name"]
            img_np = np.array(Image.open(original_img_path))

            annotator = Annotator(img_np, line_width=1)

            # Draw the boxes for user-selected classes
            for d in res["detections"]:
                box = d["box"]
                xyxy = [box["x1"], box["y1"], box["x2"], box["y2"]]
                class_idx = LABELS[d["name"]]

                annotator.box_label(
                    box=xyxy,
                    label=f"{d['name']} {d['confidence']:.2f}",
                    color=colors(class_idx, bgr=True),
                )
            annotated_img = Image.fromarray(annotator.result())

            col_img, col_counts = st.columns([3, 1])
            with col_img:
                image_zoom(annotated_img, mode="both", keep_resolution=True)
            with col_counts:
                st.table(dict(counts))
