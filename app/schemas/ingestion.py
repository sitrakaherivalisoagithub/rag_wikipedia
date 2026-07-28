from pydantic import BaseModel, Field

class WikipediaIngestion(BaseModel):
    url: str = Field(..., example="https://en.wikipedia.org/wiki/Madagascar")
    refresh: bool = Field(False, description="Set to true to delete all existing vectors before ingesting.")
