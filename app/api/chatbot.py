from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import time
import hashlib

logger = logging.getLogger("ChatbotRouter")
router = APIRouter()

# ── Request deduplication store ───────────────────────────────────────────────
_recent_requests: dict = {}
DEDUP_WINDOW_SECS = 2.0
RATE_LIMIT_WINDOW = 60.0
RATE_LIMIT_MAX    = 20

# { session_id: [timestamps] }
_rate_tracker: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session_id(request: Request, chat_request=None) -> str:
    """
    Extract session_id from request.
    Priority: body field → X-Session-ID header → client IP (fallback).
    """
    if chat_request and hasattr(chat_request, "session_id") and chat_request.session_id:
        return chat_request.session_id.strip()
    header_id = request.headers.get("X-Session-ID", "").strip()
    if header_id:
        return header_id
    client_ip = request.client.host if request.client else "unknown"
    return "ip_" + client_ip


def _is_duplicate(session_id: str, query: str) -> bool:
    key = hashlib.md5((session_id + query.strip().lower()).encode()).hexdigest()
    now = time.time()
    expired = [k for k, t in _recent_requests.items() if now - t > DEDUP_WINDOW_SECS]
    for k in expired:
        _recent_requests.pop(k, None)
    if key in _recent_requests:
        return True
    _recent_requests[key] = now
    return False


def _is_rate_limited(session_id: str) -> bool:
    now = time.time()
    if session_id not in _rate_tracker:
        _rate_tracker[session_id] = []
    _rate_tracker[session_id] = [
        t for t in _rate_tracker[session_id]
        if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_rate_tracker[session_id]) >= RATE_LIMIT_MAX:
        return True
    _rate_tracker[session_id].append(now)
    return False


def _is_rag_ready(request: Request) -> bool:
    """Check if RAG service is fully initialized and ready to accept queries."""
    try:
        svc = request.app.state.rag_service
        return svc is not None and svc.vector_db is not None
    except AttributeError:
        return False


def _error_response(message: str, session_id: str = None) -> dict:
    """
    Return a Flutter-safe error response in the same shape as ChatResponse.
    Flutter can always parse this without crashing.
    """
    return {
        "answer":       message,
        "sources":      [],
        "suggestions":  [
            {"id": 1, "text": "What animals can I see?",       "icon": "🦏"},
            {"id": 2, "text": "Tell me about Bengal tigers",   "icon": "🐯"},
            {"id": 3, "text": "What is the best time to visit?", "icon": "🕐"},
        ],
        "display_type": "text",
        "char_count":   len(message),
        "session_id":   session_id or "default",
    }


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:               str
    session_id:          Optional[str]  = None
    response_type:       Optional[str]  = "normal"
    include_suggestions: Optional[bool] = True
    use_emojis:          Optional[bool] = True


class ClearMemoryRequest(BaseModel):
    session_id: Optional[str] = None


class SuggestionChip(BaseModel):
    id:   int
    text: str
    icon: str = "🌿"


class ChatResponse(BaseModel):
    answer:       str
    sources:      List[str]            = []
    suggestions:  List[SuggestionChip] = []
    display_type: Optional[str]        = "text"
    char_count:   Optional[int]        = None
    session_id:   Optional[str]        = None


class ClearMemoryResponse(BaseModel):
    message:    str
    status:     str
    session_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Main chat endpoint.
    - Returns Flutter-safe JSON even on errors (no raw 500s)
    - Checks RAG is ready before processing
    - Deduplicates double-tap requests within 2 seconds
    - Rate limits to 20 requests/minute per session
    - Passes session_id to RAG service for isolated memory
    """
    session_id = _get_session_id(request, chat_request)

    # ── FIX 3: Check RAG is fully ready before accepting queries ─────────────
    if not _is_rag_ready(request):
        logger.warning("Query received before RAG fully initialized")
        return ChatResponse(
            **_error_response(
                "The assistant is still starting up. Please wait a few seconds and try again. 🌿",
                session_id
            )
        )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    if _is_rate_limited(session_id):
        logger.warning("Rate limit hit for session: " + session_id)
        # FIX 2: Return Flutter-safe response instead of raw 429
        return ChatResponse(
            **_error_response(
                "You're sending messages too quickly. Please wait a moment before asking again. 🙏",
                session_id
            )
        )

    # ── Deduplication ─────────────────────────────────────────────────────────
    if _is_duplicate(session_id, chat_request.query):
        logger.info("Duplicate request ignored for session: " + session_id)
        # FIX 2: Return Flutter-safe response instead of raw 429
        return ChatResponse(
            **_error_response(
                "Your message is being processed. Please wait a moment. ⏳",
                session_id
            )
        )

    try:
        rag_service = request.app.state.rag_service

        result = rag_service.query(
            message=chat_request.query,
            response_type=chat_request.response_type,
            include_suggestions=chat_request.include_suggestions,
            use_emojis=chat_request.use_emojis,
            session_id=session_id,
        )

        answer       = result.get("answer", "I couldn't find an answer for that.")
        sources      = result.get("sources", [])
        raw_sugs     = result.get("suggestions", [])
        display_type = result.get("display_type", "text")
        char_count   = result.get("char_count")

        # Convert suggestion dicts → SuggestionChip models
        suggestion_chips: List[SuggestionChip] = []
        for s in raw_sugs:
            try:
                if isinstance(s, dict):
                    suggestion_chips.append(SuggestionChip(
                        id=int(s.get("id", len(suggestion_chips) + 1)),
                        text=str(s.get("text", "")),
                        icon=str(s.get("icon", "🌿")),
                    ))
                elif isinstance(s, str) and s.strip():
                    suggestion_chips.append(SuggestionChip(
                        id=len(suggestion_chips) + 1,
                        text=s.strip(),
                        icon="🌿",
                    ))
            except Exception as e:
                logger.warning("Skipping malformed suggestion: " + str(e))

        return ChatResponse(
            answer=answer,
            sources=[str(s) for s in sources],
            suggestions=suggestion_chips,
            display_type=display_type,
            char_count=char_count,
            session_id=session_id,
        )

    except Exception as e:
        logger.error("Chat Error: " + str(e), exc_info=True)
        # FIX 2: Never return raw 500 — always return Flutter-safe response
        return ChatResponse(
            **_error_response(
                "Something went wrong on my end. Please try again. 🙏",
                session_id
            )
        )


@router.post("/chat/stream")
async def chat_stream(request: Request, chat_request: ChatRequest):
    """
    Streaming chat endpoint — text appears word-by-word on mobile.
    Better perceived performance on slow connections.
    """
    session_id = _get_session_id(request, chat_request)

    if not _is_rag_ready(request):
        async def not_ready():
            yield "Assistant is still starting up. Please try again in a moment."
        return StreamingResponse(not_ready(), media_type="text/plain")

    if _is_rate_limited(session_id):
        async def rate_limited():
            yield "Too many requests. Please wait a moment."
        return StreamingResponse(rate_limited(), media_type="text/plain")

    if _is_duplicate(session_id, chat_request.query):
        async def duplicate():
            yield "Your message is being processed."
        return StreamingResponse(duplicate(), media_type="text/plain")

    rag_service = request.app.state.rag_service

    async def token_generator():
        try:
            async for token in rag_service.query_stream(chat_request.query, session_id=session_id):
                yield token
        except Exception as e:
            logger.error("Stream error: " + str(e))
            yield "\nSomething went wrong. Please try again."

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={"X-Session-ID": session_id},
    )


@router.post("/clear-memory", response_model=ClearMemoryResponse)
async def clear_memory(request: Request, body: ClearMemoryRequest = None):
    """
    Clears conversation memory for the current session.
    FIX 1: Accepts optional body — works with or without session_id from Flutter.
    """
    try:
        # Safe session_id extraction — never crashes if body is None
        session_id = None
        if body and body.session_id:
            session_id = body.session_id.strip()
        if not session_id:
            session_id = request.headers.get("X-Session-ID", "").strip()
        if not session_id:
            client_ip  = request.client.host if request.client else "unknown"
            session_id = "ip_" + client_ip

        request.app.state.rag_service.clear_memory(session_id=session_id)
        return ClearMemoryResponse(
            message="Conversation memory cleared successfully",
            status="success",
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Memory Clear Error: " + str(e), exc_info=True)
        # FIX 2: Flutter-safe error response
        return ClearMemoryResponse(
            message="Memory clear failed — " + str(e),
            status="error",
            session_id=None,
        )


@router.get("/health")
async def health_check(request: Request):
    """
    FIX 3: Health check now verifies RAG is actually ready, not just reachable.
    Flutter should poll this until ready=True before showing the chat UI.
    """
    rag_ready = _is_rag_ready(request)
    return {
        "status":       "healthy" if rag_ready else "initializing",
        "ready":        rag_ready,       # ← Flutter checks this bool
        "timestamp":    time.time(),
        "llm_provider": "Groq",
        "llm_model":    "llama-3.1-8b-instant / llama-3.3-70b-versatile",
    }


@router.get("/status")
async def get_status(request: Request):
    """Returns full RAG engine statistics."""
    try:
        return request.app.state.rag_service.get_stats()
    except Exception:
        return {"status": "error", "message": "Service stats unavailable"}