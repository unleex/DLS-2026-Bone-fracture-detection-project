from sqlalchemy import Column, Integer, String

from database import Base


class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True)

    username = Column(String)

    input_filename = Column(String)

    output_filename = Column(String)

