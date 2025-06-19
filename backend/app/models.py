from typing import Annotated, List
import uvicorn

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from contextlib import asynccontextmanager



"""
TEAM TABLE
id, name, collection, title, people+orgs, location, description, date, subjects, accessibility
"""
class Doc(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    doc_name: str = Field(index=True)
    collection: str = Field(index=True)
    title: str = Field(index=True)
    entities: list["PeopleOrgs"] = Relationship(back_populates="people_plus_orgs")
    # index not enabled on following 2
    # location: str | None = Field(default = None)
    description: str = Field()
    date: int | None = Field(default = None, index = True)
    subject_lst: list["Subjects"] = Relationship(back_populates="subjects")
    # index not enabled
    accessibility: str | None = Field(default=None)

"""
HERO TABLES
"""
class PeopleOrgs(SQLModel, table = True):
    id: int | None = Field(default=None, primary_key=True)
    name: str | None = Field(default = None, foreign_key="doc.id")
    people_plus_orgs: Doc | None = Relationship(back_populates="entities")

class Subjects(SQLModel, table = True):
    id: int | None = Field(default=None, primary_key=True)
    name: str | None = Field(default = None, foreign_key="doc.id")
    subjects: Doc | None = Relationship(back_populates="subject_lst")



sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# to ensure a single request can use multiple threads: MAKE SURE TO SERIALIZE WRITES
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo= True, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def drop_db_and_tables():
    SQLModel.metadata.drop_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

""" 
Since Depends(get_session) yields a session for each request,
annotated is used to declare the yielded type as a class named Session of type SessionDep
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

"""
CRUD
"""

# submit data (pdf to ocr, itemize, and extract metadata) to update DB
@app.post("/heroes")
def create_db_obj(hero: Doc, session: SessionDep) -> Doc:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    # why ???????
    return hero

# retrieve data (in our case, we are retrieving metadata + PDF to serve to users)
@app.get("/heroes")
def retrieve_db_objs(session: SessionDep, offset: int, limit: Annotated[int, Query(le = 100)]) -> list[Doc]:
    heroes = session.exec(select(Doc).offset(offset).limit(limit)).all()
    return heroes

# retrieve one piece of data
@app.get("/hero/{hero_id}")
def retrieve_db_obj(hero_id: int, session: SessionDep) -> Doc:
    hero = session.exec(select(Doc).where(Doc.id == hero_id))
    if not hero:
        raise HTTPException(404, "id not found")
    return hero


@app.delete("/hero/{hero_id}")
def remove_db_objs(hero_id: int, session: SessionDep) -> Doc:
    hero = session.exec(select(Doc).where(Doc.id == hero_id))
    if not hero:
        raise HTTPException(404, "id not found")
    session.delete(hero)
    # FFLUSH / Msync type operation
    session.commit()
    return {"ok": True}




