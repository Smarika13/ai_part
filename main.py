from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import os
import logging
from dotenv import load_dotenv

# --- 1. ENVIRONMENT & LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CNP-Chatbot")

BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'

# Keep your helpful debugging prints
print("\n" + "="*60)
print("🔍 ENVIRONMENT DEBUGGING")
print("="*60)

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print("✅ .env file loaded successfully")
else:
    print(f"❌ .env file NOT FOUND at: {env_path}")

api_key = os.getenv("GOOGLE_API_KEY")
print(f"🔑 GOOGLE_API_KEY found: {api_key is not None}")
print("="*60 + "\n")

# Now import your logic files
from app.api import chatbot
from app.services.rag_service import RAGService

# Initialize RAG service instance
rag_service = RAGService()

# --- 2. LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles initialization and cleanup. 
    This is the modern way to manage shared resources.
    """
    logger.info("🚀 Starting up Chitwan National Park Chatbot API...")
    
    # Check for API Key inside lifespan to prevent silent failures
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("❌ CRITICAL: GOOGLE_API_KEY still not found!")
        raise ValueError("Missing GOOGLE_API_KEY")

    # Initialize RAG service
    rebuild = os.getenv("REBUILD_INDEX", "false").lower() == "true"
    try:
        rag_service.initialize(rebuild_index=rebuild)
        # Store in state so chatbot.py can access it via request.app.state
        app.state.rag_service = rag_service
        logger.info("✅ RAG Service initialized and attached to app state")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Service: {e}")
        raise

    yield  # --- API is running ---

    logger.info("🛑 Shutting down API...")

# --- 3. APP CONFIGURATION ---
app = FastAPI(
    title="Chitwan National Park Chatbot API",
    description="AI-powered chatbot for wildlife information",
    version="1.0.0",
    lifespan=lifespan
)

# CORS: Allows the Flutter app to connect without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router (prefix matches your Flutter baseUrl)
app.include_router(chatbot.router, prefix="/api/v1", tags=["chatbot"])

# --- 4. ENDPOINTS ---

@app.get("/", tags=["root"])
async def root():
    return {
        "message": "Chitwan National Park Chatbot API is Live! 🐯",
        "status": "running"
    }

@app.get("/api/v1/health", tags=["health"])
async def health_check_v1():
    """Matches the exact path used by Flutter ChatbotService.checkHealth()"""
    is_ready = hasattr(app.state, "rag_service")
    return {
        "status": "healthy" if is_ready else "initializing",
        "version": "1.0.0"
    }

# --- 5. EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))

    
    # Binding to 0.0.0.0 is essential for mobile device/emulator access
    uvicorn.run("main:app",
                 host="0.0.0.0",
                 port=8000,
                 reload=True)
    