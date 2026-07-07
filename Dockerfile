FROM python:3.14



COPY . .

WORKDIR /

RUN pip install -e .

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port  & streamlit run src/ui/app.py"]
