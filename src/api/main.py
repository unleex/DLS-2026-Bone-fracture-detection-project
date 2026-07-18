from ml.predictor import Predictor, AVAILABLE_MODELS

from fastapi import FastAPI

from api.schemas import QueryRequest


OUTPUT_FILE_POSTFIX = "_processed.png"

app = FastAPI()
predictor = Predictor(AVAILABLE_MODELS)


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/predict")
def predict(request: QueryRequest):

    result = predictor(
        request.filename,
        request.filename + OUTPUT_FILE_POSTFIX,
        model=request.model,
    )
    return {
        "input_filename": request.filename,
        "output_filename": request.filename + OUTPUT_FILE_POSTFIX,
        "results": result.to_json(),
    }
