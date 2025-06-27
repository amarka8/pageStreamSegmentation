# """
# TEAM TABLE
# id, name, collection, title, people+orgs, location, description, date, subjects, accessibility
# """
# class DocBase(SQLModel):
#     collection: str | None
#     title: str = Field(index=True)
#     entities: list["PeopleOrgs"] = Relationship(back_populates="people_plus_orgs")
#     # index not enabled on following 2
#     # location: str | None = Field(default = None)
#     description: str = Field()
#     date: int | None = Field(default = None, index = True)
#     subject_lst: list["Subjects"] = Relationship(back_populates="subjects")
#     # index not enabled
#     accessibility: str | None = Field(default=None)

# # Actual data model
# class Doc(DocBase, table = True):
#     id: int | None = Field(default=None, primary_key=True)
#     doc_name: str | None = Field(default=None, primary_key=True)

# # to be returned to API user
# class DocPublic(DocBase):
#     id: int

# class DocCreate(DocBase):
#     doc_name: str

# """
# HERO TABLES
# """
# class PeopleOrgsBase(SQLModel):
#     people_plus_orgs: Doc | None = Relationship(back_populates="entities")

# # Actual data model
# class PeopleOrgs(PeopleOrgsBase, table = True):
#     id: int | None = Field(default=None, primary_key=True)
#     name: str | None = Field(default = None, foreign_key="doc.doc_name")


# # to be returned to API user
# class PeopleOrgsPublic(PeopleOrgsBase):
#     id: int

# class PeopleOrgsCreate(PeopleOrgsBase):
#     name: str | None = Field(default = None, foreign_key="doc.doc_name")


# class SubjectsBase (SQLModel):
#     subjects: Doc | None = Relationship(back_populates="subject_lst")

# # Actual data model
# class Subjects(SubjectsBase, table = True):
#     id: int | None = Field(default=None, primary_key=True)
#     name: str | None = Field(default = None, foreign_key="doc.doc_name")

# # to be returned to API user
# class SubjectsPublic(SubjectsBase):
#     id: int

# class SubjectsCreate(SubjectsBase):
#     name: str | None = Field(default = None, foreign_key="doc.doc_name")