from fastapi import APIRouter, HTTPException
from app.services.schemas import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
import uuid

router = APIRouter()
rag_service = RAGService()

# Initialize the RAG system when the server starts
@router.on_event("startup")
async def startup_event():
    print("System Starting...")
    rag_service.initialize(rebuild_index=False)
    print("System Ready!")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate a new session ID if none exists
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get answer from RAG service
        result = rag_service.query(request.message)
        
        return ChatResponse(
            response=result["answer"],
            session_id=session_id,
            sources=result.get("sources", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")