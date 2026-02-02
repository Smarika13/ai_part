# AI Chatbot Service

A chatbot service powered by Google's Gemini API with RAG (Retrieval-Augmented Generation) for park information, wildlife, and activities.

## Prerequisites

- Python 3.12.0
- Google API Key (Gemini API)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Smarika13/ai_part.git
cd ai_service
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

1. Copy `.env.example` to `.env`:
```bash
   copy .env.example .env
```

2. Open `.env` and add your Google API key:
```
   GOOGLE_API_KEY=your_actual_api_key_here
```

   Get your API key from: https://makersuite.google.com/app/apikey

### 6. Run the Application
```bash
python main.py
```

The server will start at: `http://127.0.0.1:8000`

## API Documentation

Once running, visit: `http://127.0.0.1:8000/docs`

## Usage Example

**Endpoint:** `POST http://127.0.0.1:8000/api/chat`

**Request:**
```json
{
  "message": "What birds can I see?",
  "conversation_history": []
}
```

## Troubleshooting

- **API key error:** Ensure `.env` file exists with valid `GOOGLE_API_KEY`
- **Module not found:** Activate virtual environment and run `pip install -r requirements.txt`
- **Port in use:** Change `PORT` value in `.env` file

## Project Structure
```
ai_service/
├── app/
│   ├── api/          # API endpoints
│   ├── data/         # FAQ, rules, wildlife data
│   └── services/     # Core services
├── main.py           # Entry point
└── requirements.txt  # Dependencies
```