from typing import Annotated, List

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select
from contextlib import asynccontextmanager
from sqlalchemy import JSON, Column



# id, name, collection, title, people+orgs, location, description, date, subjects, accessibility
class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    collection: str = Field(index=True)
    title: str = Field(index=True)
    people_plus_orgs: List[str] = Field (index = True)
    # index not enabled on following 2
    location: List[str] | None = Field(default = None)
    description: str = Field()
    date: int | None = Field(default = None, index = True)
    subjects: List[str] = Field()
    # index not enabled
    accessibility: str | None = Field(default=None)


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

# submit data (pdf to ocr, itemize, and extract metadata)
@app.post("/heroes")
def create_db_obj(hero: Hero, session: SessionDep) -> Hero:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    # why ???????
    return hero

# retrieve data (in our case, we are retrieving metadata + PDF to serve to users)
@app.get("/heroes")
def retrieve_db_objs(session: SessionDep, offset: int, limit: Annotated[int, Query(le = 100)]) -> List[Hero]:
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes

# retrieve one piece of data


@app.get("/heroes/{hero_id}")
def retrieve_db_objs(hero_id: int, session: SessionDep) -> Hero:
    hero = session.exec(select(Hero).where(Hero.id == hero_id))
    if not hero:
        raise HTTPException(404, "id not found")
    return hero


@app.delete("/heroes/{hero_id}")
def remove_db_objs(hero_id: int, session: SessionDep) -> Hero:
    hero = session.exec(select(Hero).where(Hero.id == hero_id))
    if not hero:
        raise HTTPException(404, "id not found")
    session.delete(hero)
    # FFLUSH / Msync type operation
    session.commit()
    return {"ok": True}



