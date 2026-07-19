---
title: "Aerial object detection"
emoji: 🌆
colorFrom: blue
colorTo: purple
sdk: docker
app_file: src/api/main.py
pinned: false
---

# DLS-2026-Aerial-object-detection-project
This is a web interface for analysing images of transport and people, shot from high-altitude cameras or drones. 
It allows to easily view statistics of images including average amount of found objects, per-image object counts
and how does amount of objects change through time in your images.



## Stack
- **ML:** ![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLO-111F68?style=for-the-badge&logo=ai&logoColor=white)
- **Backend:** ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
- **Frontend:** ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
- **Deployment:** ![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=for-the-badge&logoColor=black) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## Metrics
| Class | Precision (Pro / Fast) | Recall (Pro / Fast) | mAP@50 (Pro / Fast) | mAP@50-95 (Pro / Fast) |
| --- | --- | --- | --- | --- |
| **all** | 0.647 / 0.575 | 0.559 / 0.481 | 0.572 / 0.476 | 0.359 / 0.288 |
| **pedestrian** | 0.706 / 0.630 | 0.655 / 0.577 | 0.688 / 0.594 | 0.356 / 0.292 |
| **people** | 0.660 / 0.580 | 0.524 / 0.423 | 0.543 / 0.431 | 0.243 / 0.175 |
| **bicycle** | 0.463 / 0.401 | 0.402 / 0.275 | 0.381 / 0.251 | 0.195 / 0.116 |
| **car** | 0.840 / 0.778 | 0.853 / 0.834 | 0.883 / 0.845 | 0.646 / 0.603 |
| **van** | 0.597 / 0.565 | 0.599 / 0.520 | 0.593 / 0.508 | 0.437 / 0.363 |
| **truck** | 0.624 / 0.512 | 0.525 / 0.437 | 0.531 / 0.427 | 0.374 / 0.293 |
| **tricycle** | 0.568 / 0.493 | 0.507 / 0.386 | 0.473 / 0.352 | 0.285 / 0.204 |
| **awning-tricycle** | 0.441 / 0.389 | 0.263 / 0.209 | 0.252 / 0.175 | 0.164 / 0.110 |
| **bus** | 0.842 / 0.807 | 0.659 / 0.558 | 0.719 / 0.606 | 0.549 / 0.452 |
| **motor** | 0.731 / 0.591 | 0.601 / 0.591 | 0.658 / 0.570 | 0.337 / 0.269 |

| Performance measured on Mac M1 CPU | Pro Model | Fast Model |
| --- | --- | --- |
| **Inference time per image** | 4340.6ms | 1046.5ms |

## How to Run

App is deployed here: https://huggingface.co/spaces/unleex/aerial-object-detection

~~If~~ you have hardware, that is better than the one on free version of HF Spaces, you might want to run the app locally.

### Docker
```bash
docker build -t aerial-detection .
docker run -p 7860:7860 -p 8000:8000 aerial-detection
```
### Manually from two terminals
Set up the repository:
```bash
pip install -e .
```

Terminal 1 (FastAPI Backend):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Streamlit UI):

```bash
streamlit run src/ui/app.py --server.port 7860 --server.address 0.0.0.0
```

## Adding new stats to display
Define a function that takes a list of Counters (object counts per image) and draws a pyplot, and add it to a list in src/ui/detection_stats.py. App will display it too!