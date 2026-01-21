from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chatbot
from dotenv import load_dotenv
import os

# 1. Load env vars
load_dotenv()

# 2. Define the APP object (Uvicorn looks for THIS name)
app = FastAPI(title="Chitwan National Park Chatbot API")

# 3. Add Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Include Routers
app.include_router(chatbot.router, prefix="/api/v1", tags=["chatbot"])

@app.get("/")
async def root():
    return {"message": "Chitwan API is Live!"}

# 5. Optional: Run via 'python main.py'
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)