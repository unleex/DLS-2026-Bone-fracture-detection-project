from ml.predictor import Predictor

from fastapi import FastAPI
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import Query
from schemas import QueryRequest

Base.metadata.create_all(engine)

app = FastAPI()
predictor = Predictor()

OUTPUT_FILE_POSTFIX = "_processed"


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/compute")
def compute(request: QueryRequest):

    db: Session = SessionLocal()

    row = Query(
        username=request.username,
        input_filename=request.filename,
        output_filename=request.filename + OUTPUT_FILE_POSTFIX,
    )
    predictor(row.input_filename, row.output_filename)
    db.add(row)
    db.commit()
    db.refresh(row)
    db.close()

    return {
        "id": row.id,
        "username": row.username,
        "input_filename": row.input_filename,
        "output_filename": row.output_filename,
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
