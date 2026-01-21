# Chitwan Wildlife AI Service (RAG) 🐾

This is the backend service for the Chitwan National Park AI Assistant. It uses a **RAG (Retrieval-Augmented Generation)** system to answer questions based on local wildlife data and park regulations.

## 📁 Project Structure
- `wildlife/`: Contains JSON files for Amphibians, Birds, Mammals, etc.
- `app/data/raw/`: Contains `faq.txt` and `activities.json`.
- `app/services/rag_service.py`: The core AI logic.
- `vector_store/`: (Auto-generated) Stores the FAISS search index.

## 🚀 Setup Instructions for Friends

### 1. Clone and Navigate
```bash
git clone [https://github.com/Smarika13/ai_part.git](https://github.com/Smarika13/ai_part.git)
cd ai_part/ai_service
