from pydantic import Field
from src.schemas.entities import InputModel
from src.constants.domain import MAX_CHARACTERS, MAX_DOCUMENTS, MAX_CSV_CHARACTERS

class ProjectionDocument(InputModel):
    doc_id:str=Field(min_length=1,max_length=200)
    text:str=Field(min_length=1,max_length=MAX_CHARACTERS)

class ProjectionInput(InputModel):
    documents:list[ProjectionDocument]|None=Field(default=None,max_length=MAX_DOCUMENTS)
    csv_text:str|None=Field(default=None,min_length=1,max_length=MAX_CSV_CHARACTERS)
