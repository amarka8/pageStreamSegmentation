from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from contextlib import asynccontextmanager


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    collection: str = Field(index=True)
    title: str = Field(index=True)
    people_plus_orgs: list[str] = Field (index = True)
    # index not enabled on following 2
    location: list[str] | None = Field(default = None)
    description: str = Field()
    date: int | None = Field(default = None, index = True)
    subjects: list[str] = Field(index = True)
    # index not enabled
    accessibility: str | None = Field(default=None)

# Code above omitted 👆

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# to ensure a single request can use multiple threads: MAKE SURE TO SERIALIZE WRITES
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

""" 
Since Depends(get_session) yields a session for each request,
annotated is used to declare the yielded type as a class named Session
"""

SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # executed before app starts taking requests
    create_db_and_tables()
    # add code for when you are stopping the application
    yield

# for development - change to use migrations eventually
app = FastAPI(lifespan=lifespan)


