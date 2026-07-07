FROM python:3.14



COPY . .

WORKDIR /

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "src/api/main:app", "--host", "0.0.0.0", "--port", "7860"]
