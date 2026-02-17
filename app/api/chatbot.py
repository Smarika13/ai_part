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
    response_type: Optional[str] = "normal"  # ✨ NEW: "short" or "normal"
    include_suggestions: Optional[bool] = True
    use_emojis: Optional[bool] = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]  # Explicitly defined as List of Strings
    suggestions: List[str] = []  # Defaults to empty list to prevent Flutter null errors

class ClearMemoryResponse(BaseModel):
    message: str
    status: str

# --- Endpoints ---

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Main Chat endpoint. Returns Markdown-formatted answers, 
    source citations, and clickable suggestion chips.
    
    NEW: Supports response_type parameter:
    - "short": 2-3 sentence responses
    - "normal": Detailed responses (default)
    """
    try:
        # Access the globally initialized RAG service from main.py
        rag_service = request.app.state.rag_service
        
        # ✨ UPDATED: Pass response_type to RAG service
        result = rag_service.query(
            message=chat_request.query,
            response_type=chat_request.response_type,  # ✨ NEW
            include_suggestions=chat_request.include_suggestions,
            use_emojis=chat_request.use_emojis
        )

        # Extract and sanitize data
        answer = result.get("answer", "I'm sorry, I couldn't find an answer for that. 🐯")
        sources = result.get("sources", [])
        suggestions = result.get("suggestions", [])

        # ✅ Return clean response with suggestions as separate list
        return ChatResponse(
            answer=answer,  # Clean answer WITHOUT suggestion text
            sources=[str(s) for s in sources],
            suggestions=[str(s) for s in suggestions]  # Suggestions as list
        )

    except Exception as e:
        logger.error(f"Chat Error: {str(e)}")
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