from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    include_suggestions: Optional[bool] = True  # ← NEW: Optional flag for suggestions
    use_emojis: Optional[bool] = True  # ← NEW: Optional flag for emojis

class ChatResponse(BaseModel):
    answer: str
    sources: list
    suggestions: Optional[List[str]] = []  # ← NEW: Add suggestions to response

class ClearMemoryResponse(BaseModel):
    message: str
    status: str

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat endpoint with smart suggestions and emoji formatting
    """
    try:
        rag_service = request.app.state.rag_service
        
        # Query with suggestions and emoji support
        result = rag_service.query(
            chat_request.query, 
            include_suggestions=chat_request.include_suggestions,
            use_emojis=chat_request.use_emojis
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            suggestions=result.get("suggestions", [])  # ← NEW: Include suggestions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/clear-memory")
async def clear_memory(request: Request):
    """
    Clear conversation memory - start a new conversation
    """
    try:
        rag_service = request.app.state.rag_service
        rag_service.clear_memory()
        
        return ClearMemoryResponse(
            message="Conversation memory cleared successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/chat-history")
async def get_chat_history(request: Request):
    """
    Get current conversation history
    """
    try:
        rag_service = request.app.state.rag_service
        history = rag_service.get_chat_history()
        
        return {
            "conversation_turns": len(history),
            "history": [
                {
                    "role": msg.type if hasattr(msg, 'type') else 'unknown',
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                }
                for msg in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/status")
async def status(request: Request):
    """
    Get RAG service statistics
    """
    return request.app.state.rag_service.get_stats()