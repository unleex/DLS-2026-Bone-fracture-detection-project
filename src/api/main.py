from ml.predictor import Predictor

from fastapi import FastAPI
from sqlalchemy.orm import Session

from database.database import Base, SessionLocal, engine
from api.models import Query
from api.schemas import QueryRequest

Base.metadata.create_all(engine)

OUTPUT_FILE_POSTFIX = "_processed"
AVAILABLE_MODELS = {"Pro": "runs/detect/train-10/weights/best.pt"}

app = FastAPI()
predictor = Predictor(AVAILABLE_MODELS)


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/predict")
def predict(request: QueryRequest):

    db: Session = SessionLocal()

    row = Query(
        input_filename=request.filename,
        output_filename=request.filename + OUTPUT_FILE_POSTFIX,
    )
    result = predictor(
        row.input_filename,
        row.output_filename,
        model=request.model,
        classes=request.classes,
    )
    detected = [result.names[idx] for idx in result.boxes.cls]

    db.add(row)
    db.commit()
    db.refresh(row)
    db.close()

    return {
        "id": row.id,
        "input_filename": row.input_filename,
        "output_filename": row.output_filename,
        "detected": detected,
    }


@app.get("/history")
def global_history():

    db = SessionLocal()

    rows = db.query(Query).all()

    db.close()

    return rows


@app.get("/history/{username}")
def user_history(username: str):

    db = SessionLocal()

    rows = db.query(Query).filter(Query.username == username).all()

    db.close()

    return rows
