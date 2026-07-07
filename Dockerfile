FROM python:3.14



COPY . .

WORKDIR /

RUN pip install -e .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 7860  & streamlit run src/ui/app.py"]
