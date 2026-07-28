"""
Main entry point for the FastAPI chat application.

This module initializes the FastAPI application, sets up the connection to the
database, configures the LangGraph workflow, and defines the chat endpoint.
"""

import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
import uvicorn

from app.core.agent_manager import AgentManager
from app.schemas.chat import ChatRequest, MessageResponse
from app.schemas.ingestion import WikipediaIngestion
from app.services.ingestion_flow import ingest_wikipedia_url, insert_documents_in_vector_store
from app.db.vector_store import VectorStoreManager
from app.core.dependencies import vector_store
from fastapi import HTTPException, status

# --- Load Environment Variables ---
load_dotenv()

agent_manager = AgentManager()



# --- FastAPI Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events for startup and shutdown.
    
    During startup, it initializes Vertex AI, creates a database connection pool,
    sets up the LangGraph checkpointer, and compiles the graph.
    The connection pool is automatically closed on shutdown.
    """
    
    await agent_manager.setup()

    yield

# --- FastAPI Application ---

api = FastAPI(lifespan=lifespan)

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests by invoking the LangGraph workflow.
    
    Args:
        request: A ChatRequest object containing the user's message and thread_id.
        
    Returns:
        A dictionary containing the AI's response, the thread_id, and any sources.
    """
    # 1. Get or create a thread ID
    thread_id = request.thread_id or str(uuid.uuid4())
    print(f'\nRequest for thread "{thread_id}": {request.message}\n')

    # 2. Get the current state of the conversation
    current_state = await agent_manager.get_state(thread_id)
    print(f"Current state: {current_state}")
    
    # 3. Add the new user message to the history
    messages = current_state.get("messages", [])
    messages.append(HumanMessage(content=request.message))

    # 4. Invoke the graph with the updated message history
    inputs = {"messages": messages, "context": []}
    result = await agent_manager.invoke(inputs, thread_id)
    print(f"Final state: {result}")

    # 5. Extract the last AI response
    last_message = result["messages"][-1]

    # 6. Format sources to include page numbers if available
    sources = []
    for doc in result.get("context", []):
        source_info = {
            "source": doc.metadata.get("source"),
            "section": doc.metadata.get("h2_section"),
            "sub_section": doc.metadata.get("h3_section"),
            "type": doc.metadata.get("content_type")
        }
        sources.append(source_info)

    return MessageResponse(
        response=last_message.content,
        thread_id=thread_id,
        language=result.get("language"),
        sources=sources
    )

@api.post("/add", status_code=status.HTTP_201_CREATED)
async def add_document(item: WikipediaIngestion):
    try:
        db_manager = VectorStoreManager(vector_store)

        if item.refresh:
            db_manager.delete_all_documents()

        documents = ingest_wikipedia_url(item.url)
        
        if not documents:
            return {
                "status": "failed_or_empty",
                "message": "Le document a été lu mais ne contenait aucune donnée exploitable.",
                "inserted_ids": []
            }
        
        inserted_ids = insert_documents_in_vector_store(documents)
            
        return {
            "status": "success",
            "message": f"Document traité avec succès. {len(inserted_ids)} chunks insérés.",
            "inserted_ids": inserted_ids
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur interne lors du traitement de l'ingestion: {str(e)}"
        )

# --- Main Execution Block ---

if __name__ == "__main__":
    # This block is for running the application locally for development.
    # It uses uvicorn to serve the FastAPI application.
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)
