from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: str = None 


class MessageResponse(BaseModel):
    response: str
    thread_id: str
    language: str
    sources: list[dict] = []
