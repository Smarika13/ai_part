from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

from app.api import chatbot
from app.services.rag_service import RAGService

# Load environment variables from .env
load_dotenv()

# Initialize RAG service globally (do not call initialize yet)
rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown
    """
    # Startup
    print("🚀 Starting up Chitwan National Park Chatbot API...")

    # Determine whether to rebuild index from env
    rebuild = os.getenv("REBUILD_INDEX", "false").lower() == "true"

    # Initialize RAG service
    try:
        rag_service.initialize(rebuild_index=rebuild)
        # Store in app state for routers
        app.state.rag_service = rag_service
        print("✅ RAG Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG Service: {e}")
        raise

    yield  # Application is running

    # Shutdown
    print("🛑 Shutting down Chitwan National Park Chatbot API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Chitwan National Park Chatbot API",
    description="AI-powered chatbot for wildlife information and park activities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chatbot.router, prefix="/api/v1", tags=["chatbot"])

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Chitwan National Park Chatbot API is Live! 🐯",
        "version": "1.0.0",
        "status": "running"
    }

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    rag_stats = getattr(app.state, "rag_service", None)
    stats = rag_stats.get_stats() if rag_stats else {}
    return {
        "status": "healthy" if rag_stats else "rag_service not initialized",
        "rag_service": stats
    }

# Run with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
