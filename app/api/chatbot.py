from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

# Set up logging for professional tracking
logger = logging.getLogger("ChatbotRouter")

router = APIRouter()

# --- Request/Response Models ---

class ChatRequest(BaseModel):
    query: str
    include_suggestions: Optional[bool] = True
    use_emojis: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]  # Explicitly defined as List of Strings
    suggestions: List[str] = [] # Defaults to empty list to prevent Flutter null errors

class ClearMemoryResponse(BaseModel):
    message: str
    status: str

# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Main Chat endpoint. Returns Markdown-formatted answers, 
    source citations, and clickable suggestion chips.
    """
    try:
        # Access the globally initialized RAG service from main.py
        rag_service = request.app.state.rag_service
        
        # 1. Execute the query via RAG Service
        # Ensure your rag_service.query returns a dictionary with these keys
        result = rag_service.query(
            chat_request.query, 
            include_suggestions=chat_request.include_suggestions,
            use_emojis=chat_request.use_emojis
        )

        # 2. Extract and sanitize data
        # We use .get() with defaults to prevent KeyErrors
        answer = result.get("answer", "I'm sorry, I couldn't find an answer for that. 🐯")
        sources = result.get("sources", [])
        suggestions = result.get("suggestions", [])

        # 3. Return the structured response
        return ChatResponse(
            answer=answer,
            sources=[str(s) for s in sources], # Ensure all sources are strings
            suggestions=[str(s) for s in suggestions] # Ensure all suggestions are strings
        )

    except Exception as e:
        logger.error(f"Chat Error: {str(e)}")
        # Return a 500 error that the Flutter App's try-catch will handle
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/clear-memory", response_model=ClearMemoryResponse)
async def clear_memory(request: Request):
    """
    Clears the conversation buffer for the current session.
    """
    try:
        rag_service = request.app.state.rag_service
        rag_service.clear_memory()
        
        return ClearMemoryResponse(
            message="Conversation memory cleared successfully",
            status="success"
        )
    except Exception as e:
        logger.error(f"Memory Clear Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear memory")

@router.get("/status")
async def get_status(request: Request):
    """
    Returns the current health and statistics of the RAG engine.
    """
    try:
        return request.app.state.rag_service.get_stats()
    except Exception:
        return {"status": "error", "message": "Service stats unavailable"}