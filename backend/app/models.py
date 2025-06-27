from typing import Annotated, List
import uvicorn

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from contextlib import asynccontextmanager

    

"""
TEAM TABLE
id, name, collection, title, people+orgs, location, description, date, subjects, accessibility
"""
class DocBase(SQLModel):
    collection: str | None = Field(index = True, default= None)
    entities: str | None = Field(default=None) # separated by , or |
    # index not enabled on following 2
    # location: str | None = Field(default = None)
    description: str | None = Field(default=None)
    date: str | None = Field(default = None, index = True)  # expect 'YYYYMM'
    subject_lst: str | None = Field(default=None) # separated by , or |
    # index not enabled
    accessibility: str | None = Field(default=None)

# Actual data model
class Doc(DocBase, table = True):
    id: int | None = Field(default=None, primary_key=True)
    doc_name: str 

# to be returned to API user
class DocPublic(DocBase):
    id: int

class DocCreate(DocBase):
    doc_name: str


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# to ensure a single request can use multiple threads: MAKE SURE TO SERIALIZE WRITES
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo= True, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # SQLModel.metadata.sess

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
@app.post("/doc/", response_model=DocPublic)
def create_db_obj(doc: DocCreate, session: SessionDep):
    db_doc = Doc.model_validate(doc)
    session.add(db_doc)
    session.commit()
    session.refresh(db_doc)
    # why ???????
    return db_doc


# retrieve data (in our case, we are retrieving metadata + PDF to serve to users)
@app.get("/docs/", response_model= list[DocPublic])
def retrieve_db_objs(session: SessionDep, offset: int, limit: Annotated[int, Query(le = 100)]):
    docs = session.exec(select(Doc).offset(offset).limit(limit)).all()
    return docs

# retrieve one piece of data
@app.get("/doc/{doc_id}", response_model=DocPublic)
def retrieve_db_obj(doc_id: int, session: SessionDep):
    doc = session.exec(select(Doc).where(Doc.id == doc_id))
    if not doc:
        raise HTTPException(404, "id not found")
    return doc


@app.delete("/doc/{doc_id}")
def remove_db_objs(doc_id: int, session: SessionDep):
    doc = session.exec(select(Doc).where(Doc.id == doc_id))
    if not doc:
        raise HTTPException(404, "id not found")
    session.delete(doc)
    # FFLUSH / Msync type operation
    session.commit()
    return {"ok": True}




