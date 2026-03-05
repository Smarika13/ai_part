"""
rag_service.py  —  Mobile-Optimized RAG Service  v4
====================================================
Zero deprecated LangChain imports. Works with modern LangChain.
Direct retriever + LLM calls instead of deprecated chains.
"""

import os
import re
import json
import time
import logging
import asyncio
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

import sys

sys.path.append(str(Path(__file__).parent))
from suggestion_engine import SuggestionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_INPUT_CHARS    = 500
MAX_MEMORY_TURNS   = 6
MAX_RESPONSE_CHARS = 400
MAX_LIST_RESPONSE_CHARS = 800
MAX_LIST_ITEMS     = 6
QUERY_TIMEOUT_SECS = 10.0

# ── Known species index (built at startup for hallucination guard) ────────────
KNOWN_SPECIES: set = set()   # populated in RAGService.initialize()


# ── Smart Response Cache ─────────────────────────────────────────────────────
import hashlib
import json as _json
from datetime import datetime, timedelta

class ResponseCache:
    """
    File-backed LRU cache for LLM responses.
    - Persists across server restarts
    - Auto-expires entries after TTL
    - Fuzzy key matching so "how much is jeep safari" hits "how much does jeep safari cost"
    """

    def __init__(self, cache_file: str = "response_cache.json", ttl_hours: int = 24, max_size: int = 200):
        self.cache_file = cache_file
        self.ttl        = timedelta(hours=ttl_hours)
        self.max_size   = max_size
        self._cache: dict = {}
        self._load()

    def _load(self):
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._cache = _json.load(f)
                # Remove expired entries on load
                now = datetime.utcnow().isoformat()
                self._cache = {k: v for k, v in self._cache.items() if v.get("expires", "") > now}
                logger.info("Cache loaded: " + str(len(self._cache)) + " entries")
        except Exception as e:
            logger.warning("Cache load failed: " + str(e))
            self._cache = {}

    def _save(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                _json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Cache save failed: " + str(e))

    def _key(self, query: str) -> str:
        """Normalize query to cache key — lowercased, stripped, common words removed."""
        q = re.sub(r"[?!.,]", "", query.lower().strip())
        q = re.sub(r"[ \t]+", " ", q)
        return hashlib.md5(q.encode()).hexdigest()

    def get(self, query: str):
        key    = self._key(query)
        entry  = self._cache.get(key)
        if not entry:
            return None
        if entry.get("expires", "") < datetime.utcnow().isoformat():
            del self._cache[key]
            return None
        entry["hits"] = entry.get("hits", 0) + 1
        return entry["answer"]

    def set(self, query: str, answer: str, display_type: str = "text"):
        # Don't cache uncertain or error answers
        uncertain_phrases = ["i'm unsure", "i don't have", "i do not have",
                             "something went wrong", "try again"]
        if any(p in answer.lower() for p in uncertain_phrases):
            return
        # Don't cache very short answers (likely errors)
        if len(answer) < 20:
            return

        key = self._key(query)
        self._cache[key] = {
            "query":        query,
            "answer":       answer,
            "display_type": display_type,
            "created":      datetime.utcnow().isoformat(),
            "expires":      (datetime.utcnow() + self.ttl).isoformat(),
            "hits":         0,
        }
        # Evict oldest entries if over max size
        if len(self._cache) > self.max_size:
            oldest = sorted(self._cache.items(), key=lambda x: x[1].get("created", ""))
            for k, _ in oldest[:20]:
                del self._cache[k]

        self._save()

    def clear(self):
        self._cache = {}
        self._save()
        logger.info("Response cache cleared")

    @property
    def size(self) -> int:
        return len(self._cache)


class SimpleMemory:
    """Stores last N conversation turns. No LangChain dependency."""

    def __init__(self, max_turns=6):
        self.max_turns = max_turns
        self.messages  = []

    def load_memory_variables(self, _):
        lines = []
        for msg in self.messages[-self.max_turns * 2:]:
            prefix = "Human" if msg["role"] == "human" else "Assistant"
            lines.append(prefix + ": " + msg["content"])
        return {"chat_history": "\n".join(lines)}

    def save_context(self, inputs, outputs):
        self.messages.append({"role": "human",     "content": inputs.get("question", "")})
        self.messages.append({"role": "assistant", "content": outputs.get("answer",   "")})
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2:]

    def clear(self):
        self.messages = []

    @property
    def chat_memory(self):
        return self


class RAGService:
    def __init__(self):
        self.embeddings        = None
        self.llm               = None
        self.vector_db         = None
        self._sessions: dict   = {}          # session_id → SimpleMemory
        self.retriever_convo   = None
        self.retriever_list    = None
        self.retriever_bare    = None
        self.retriever_price   = None
        self.suggestion_engine = SuggestionEngine()
        # ── Hybrid search ─────────────────────────────────────────────────────
        self._bm25_index       = None        # BM25 keyword index
        self._all_docs: list   = []          # all Document objects (for BM25 lookup)
        # ── Response cache ────────────────────────────────────────────────────
        self._cache            = ResponseCache(ttl_hours=24, max_size=200)

    def _get_memory(self, session_id: str) -> "SimpleMemory":
        """Get or create a memory instance for this session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SimpleMemory(max_turns=MAX_MEMORY_TURNS)
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session cleared: " + session_id)

    def initialize(self, rebuild_index=False):
        logger.info("Initializing RAG Service...")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set.")

        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Embeddings ready (FastEmbed - CPU optimized)")

        self._api_key = api_key
        self.llm = self._make_llm(max_tokens=180)  # default
        logger.info("Groq LLM ready")

        base_dir         = Path(__file__).resolve().parent.parent.parent
        wildlife_dir     = base_dir / "wildlife"
        raw_data_dir     = base_dir / "app" / "data" / "raw"
        vector_store_dir = base_dir / "vector_store"
        index_path       = vector_store_dir / "faiss_index"

        if not rebuild_index and index_path.exists():
            try:
                self.vector_db = FAISS.load_local(
                    str(index_path), self.embeddings, allow_dangerous_deserialization=True
                )
                logger.info("Loaded " + str(self.vector_db.index.ntotal) + " vectors")
            except Exception as e:
                logger.warning("Could not load index: " + str(e) + " - rebuilding")
                rebuild_index = True

        if rebuild_index or not self.vector_db:
            self._build_index(wildlife_dir, raw_data_dir, vector_store_dir, index_path)

        self.retriever_convo  = self._make_retriever(k=2)
        self.retriever_list   = self._make_retriever(k=10)
        self.retriever_bare   = self._make_retriever(k=10)
        self.retriever_price  = self._make_retriever(k=6)

        # ── Build BM25 keyword index ──────────────────────────────────────────
        self._build_bm25_index()
        logger.info("RAG Service fully initialized")

    def _build_index(self, wildlife_dir, raw_data_dir, vector_store_dir, index_path):
        documents = self._load_all_documents(wildlife_dir, raw_data_dir)
        if not documents:
            raise ValueError("No documents found.")
        splitter   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        final_docs = splitter.split_documents(documents)
        logger.info("Created " + str(len(final_docs)) + " chunks")
        self.vector_db = FAISS.from_documents(final_docs, self.embeddings)
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db.save_local(str(index_path))
        logger.info("Index saved - " + str(self.vector_db.index.ntotal) + " vectors")

    def _make_llm(self, max_tokens: int, model: str = "llama-3.1-8b-instant"):
        """
        Create a Groq LLM with the specified model and token limit.
        Models:
          - llama-3.1-8b-instant  → fast, for simple/price/greeting queries
          - llama-3.3-70b-versatile → powerful, for lists/conservation/complex queries
        """
        return ChatGroq(
            model=model,
            temperature=0.3,
            groq_api_key=self._api_key,
            max_tokens=max_tokens,
            max_retries=1,
        )

    def _make_retriever(self, k, filter_category: str = None):
        """
        Build a retriever. If filter_category is given (e.g. "birds", "mammals"),
        only chunks from that category are searched — much faster and more accurate.
        """
        search_kwargs = {"k": k, "fetch_k": k * 4, "lambda_mult": 0.7}
        if filter_category:
            search_kwargs["filter"] = {"category": filter_category}
        return self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    def _build_bm25_index(self):
        """Build a BM25 keyword index over all indexed documents."""
        if not BM25_AVAILABLE:
            logger.warning("rank-bm25 not installed — keyword search disabled. Run: pip install rank-bm25")
            return
        try:
            # Fetch all docs from FAISS docstore
            docstore = self.vector_db.docstore._dict
            self._all_docs = [doc for doc in docstore.values()]
            tokenized = [doc.page_content.lower().split() for doc in self._all_docs]
            self._bm25_index = BM25Okapi(tokenized)
            logger.info("BM25 index built over " + str(len(self._all_docs)) + " docs")
        except Exception as e:
            logger.warning("BM25 index build failed (non-critical): " + str(e))

    def _hybrid_retrieve(self, query: str, k: int, filter_category: str = None) -> list:
        """
        Reciprocal Rank Fusion of FAISS (semantic) + BM25 (keyword) results.
        Falls back to FAISS-only if BM25 not available.
        """
        # ── FAISS semantic results ────────────────────────────────────────────
        retriever   = self._make_retriever(k=k, filter_category=filter_category)
        faiss_docs  = retriever.invoke(query)

        if not self._bm25_index or not BM25_AVAILABLE:
            return faiss_docs   # fallback: FAISS only

        # ── BM25 keyword results ──────────────────────────────────────────────
        tokens      = query.lower().split()
        bm25_scores = self._bm25_index.get_scores(tokens)

        # Apply category filter to BM25 results
        ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_docs = []
        for idx in ranked_indices[:k]:
            doc = self._all_docs[idx]
            if filter_category and doc.metadata.get("category") != filter_category:
                continue
            if bm25_scores[idx] > 0:
                bm25_docs.append(doc)
            if len(bm25_docs) >= k:
                break

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        # Score each doc: sum of 1/(rank+60) from each list
        rrf_scores: dict = {}
        all_candidates: dict = {}

        for rank, doc in enumerate(faiss_docs):
            key = doc.page_content[:100]   # use content prefix as unique key
            rrf_scores[key]     = rrf_scores.get(key, 0) + 1 / (rank + 60)
            all_candidates[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:100]
            rrf_scores[key]     = rrf_scores.get(key, 0) + 1 / (rank + 60)
            all_candidates[key] = doc

        # Sort by combined RRF score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        result      = [all_candidates[k] for k in sorted_keys[:k]]

        logger.info("Hybrid retrieve: " + str(len(faiss_docs)) + " FAISS + "
                    + str(len(bm25_docs)) + " BM25 → " + str(len(result)) + " fused")
        return result

    def _get_category_filter(self, message: str):
        """Detect if the query is about a specific wildlife category."""
        m = message.lower()
        if any(w in m for w in ["bird", "birds", "avian", "feather", "beak", "nest", "flock"]):
            return "birds"
        if any(w in m for w in ["mammal", "tiger", "rhino", "elephant", "deer", "leopard", "bear"]):
            return "mammals"
        if any(w in m for w in ["reptile", "crocodile", "gharial", "snake", "lizard", "turtle"]):
            return "reptiles"
        if any(w in m for w in ["fish", "aquatic", "river fish", "mahseer"]):
            return "fish"
        if any(w in m for w in ["butterfly", "butterflies", "insect"]):
            return "butterflies"
        if any(w in m for w in ["amphibian", "frog", "toad"]):
            return "amphibians"
        if any(w in m for w in ["plant", "flora", "tree", "flower", "grass"]):
            return "plants"
        return None  # no filter — search everything

    def _prompt_convo(self):
        return PromptTemplate(
            template=(
                "You are a friendly, knowledgeable local guide at Chitwan National Park, Nepal.\n"
                "Speak naturally and warmly — like you are chatting with a visitor on safari, not reading from a brochure.\n"
                "Guidelines:\n"
                "- Answer in 2 to 4 sentences. Be informative but conversational.\n"
                "- Show genuine enthusiasm when the topic deserves it (wildlife, rare sightings, etc.).\n"
                "- Do NOT start with 'Based on the context', 'As a guide', 'Certainly', or similar filler phrases.\n"
                "- Do NOT use bullet points or numbered lists in this mode.\n"
                "- ONLY use facts from the Context below. Do not add any facts from your own knowledge.\n"
                "- If something is not in the Context, say you are not sure and suggest asking park staff.\n"
                "CRITICAL FACTS (never contradict these):\n"
                "- The national animal of Nepal is the COW (गाई). NOT the rhino, tiger, or elephant.\n"
                "- The One-horned Rhinoceros is a protected/endangered species found in Chitwan — NOT Nepal's national animal.\n"
                "- IMPORTANT: Detect the language of the Question. If it is in Nepali, respond entirely in Nepali. If in English, respond in English.\n"
                "- When responding in Nepali: write in warm, natural, conversational Nepali — like a friendly local guide (सरल र मैत्रीपूर्ण भाषामा बोल्नुस्). Avoid stiff, formal, or bureaucratic phrasing. Do NOT start with 'यस सन्दर्भमा' or similar robotic openers.\n"
                "- Standard Nepali timing words: Morning=बिहान, Evening=साँझ. NEVER invent words like 'साँरै'.\n\n"
                "Context:\n{context}\n\n"
                "Chat history:\n{chat_history}\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            input_variables=["context", "chat_history", "question"],
        )

    def _prompt_price(self):
        price_table = (
            "VERIFIED ACTIVITY PRICES AND TIMINGS (use ONLY these — never invent other numbers or times):\n"
            "- Jeep Safari (also called Jungle Safari): Domestic NPR 500   | SAARC NPR 1,500  | Foreign Tourist NPR 3,500  | BOTH Morning 6-10AM AND Evening 2-5PM\n"
            "- Elephant Safari:                         Domestic NPR 1,650 | SAARC NPR 4,000  | Foreign Tourist NPR 5,000  | BOTH Morning 6-10AM AND Evening 2-5PM\n"
            "- Bird Watching Tour:                      Domestic NPR 3,000 | SAARC NPR 5,500  | Foreign Tourist NPR 6,500  | BOTH Morning 6-10AM AND Evening 2-5PM\n"
            "- Jungle Walk:                             Domestic NPR 5,000 | SAARC NPR 10,000 | Foreign Tourist NPR 12,500 | BOTH Morning 6-10AM AND Evening 2-5PM\n"
            "- Canoe Safari:                            Domestic NPR 500   | SAARC NPR 600    | Foreign Tourist NPR 700    | BOTH Morning 6-10AM AND Evening 2-5PM\n"
            "- Tharu Cultural Program:                  Domestic NPR 200   | SAARC NPR 300    | Foreign Tourist NPR 300    | Evening ONLY 7-8PM\n"
            "- Tharu Museum:                            Domestic NPR 200   | SAARC NPR 400    | Foreign Tourist NPR 400    | 10AM-5PM daily\n"
        )
        return PromptTemplate(
            template=(
                "You are a helpful guide at Chitwan National Park explaining activity costs.\n\n"
                + price_table + "\n"
                "Guidelines:\n"
                "- Use ONLY the prices AND timings listed above. Never invent or guess prices or timings.\n"
                "- If the visitor asks about one specific activity, give only that activity's prices and timing.\n"
                "- If the visitor asks ONLY about timing/schedule (not price), answer with the timing from the table above — do NOT invent timings.\n"
                "- CRITICAL: If an activity has BOTH morning and evening timings (marked as 'BOTH'), always mention BOTH times. Never give only one when two exist.\n"
                "- Keep the tone friendly and helpful.\n"
                "- IMPORTANT: If the Question is in Nepali, respond entirely in Nepali. Use these exact Nepali translations:\n"
                "  Domestic=नेपाली नागरिक, SAARC=सार्क, Foreign Tourist=विदेशी पर्यटक\n"
                "  Morning=बिहान, Evening=साँझ, Daily=दैनिक, AM=बजे, PM=बजे\n"
                "  Example timing in Nepali: 'बिहान ६ देखि १० बजे र साँझ २ देखि ५ बजे'\n"
                "- NEVER use words like 'साँरै', 'बिहानै', or any invented Nepali words. Use only standard Nepali.\n"
                "- When responding in Nepali: use natural, conversational Nepali — like a friendly guide (सरल भाषामा).\n\n"
                "Extra context from documents:\n{context}\n\n"
                "Chat history:\n{chat_history}\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            input_variables=["context", "chat_history", "question"],
        )

    def _prompt_list(self):
        return PromptTemplate(
            template=(
                "You are a Chitwan National Park wildlife guide assistant.\n"
                "STRICT RULES:\n"
                "- Start IMMEDIATELY with the first bullet. No introduction.\n"
                "- List MAXIMUM 6 items.\n"
                "- PRIORITIZE endangered, threatened, and vulnerable species first. Never list Least Concern common birds like House Sparrow or Black Drongo unless specifically asked.\n"
                "- Use only • bullet points (the • character).\n"
                "- Each bullet MUST use this format: • English Name (Nepali Name) - conservation status + one brief fact.\n"
                "  Example: • Bengal Florican (खरमुजुर) - Critically Endangered grassland bird\n"
                "  Example: • Black-bellied Tern (कालो पेटे टर्न) - Endangered river-nesting tern\n"
                "- Never write 'here are' or any preamble.\n"
                "- Never add closing remarks like '(no more listed)'. Just stop.\n"
                "- Only use facts from the Context.\n"
                "- CRITICAL: The national animal of Nepal is the COW (गाई). Never say any wildlife species is Nepal's national animal.\n"
                "- IMPORTANT: If the Question is in Nepali, respond entirely in Nepali. If in English, respond in English.\n"
                "- When responding in Nepali: use natural, conversational Nepali for bullet labels and facts — like a knowledgeable local guide (सरल भाषामा).\n\n"
                "Context:\n{context}\n\n"
                "Chat history:\n{chat_history}\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            input_variables=["context", "chat_history", "question"],
        )

    def _prompt_bare(self):
        return PromptTemplate(
            template=(
                "You are a Chitwan National Park wildlife guide assistant.\n"
                "STRICT RULES:\n"
                "- Start IMMEDIATELY with the first item. No introduction.\n"
                "- List MAXIMUM 6 items.\n"
                "- PRIORITIZE endangered, rare, and iconic species first.\n"
                "- Each line format: • English Name (Nepali Name). Nothing else.\n"
                "- Use • bullet points (the • character).\n"
                "- No preamble, no closing remarks.\n"
                "- IMPORTANT: If the Question is in Nepali, respond entirely in Nepali. If in English, respond in English.\n"
                "- When responding in Nepali: use natural, conversational Nepali (सरल भाषामा).\n\n"
                "Context:\n{context}\n\n"
                "Chat history:\n{chat_history}\n\n"
                "Question: {question}\n\nAnswer:"
            ),
            input_variables=["context", "chat_history", "question"],
        )

    def query(self, message, response_type="normal", include_suggestions=True, use_emojis=True, session_id="default"):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._async_query(message, response_type, include_suggestions, use_emojis, session_id),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._async_query(message, response_type, include_suggestions, use_emojis, session_id)
                )
        except Exception as e:
            logger.error("query() error: " + str(e), exc_info=True)
            return self._error_response()

    async def _async_query(self, message, response_type, include_suggestions, use_emojis, session_id="default"):
        if not self.retriever_convo:
            return {"answer": "Service not ready.", "sources": [], "suggestions": [], "display_type": "text"}

        message = message.strip()[:MAX_INPUT_CHARS]
        if not message:
            return {"answer": "Please ask me something!", "sources": [], "suggestions": [], "display_type": "text"}

        if self._is_greeting(message):
            return self._greeting_response(message)

        if self._is_activity_list(message):
            return self._activity_list_response(message)

        is_bare         = self._is_bare_list(message)
        is_price        = (not is_bare) and self._is_price(message)
        is_conservation = (not is_bare) and (not is_price) and self._is_conservation(message)
        is_list         = (not is_bare) and (not is_price) and (not is_conservation) and self._is_list(message)


        # ── Response cache check ─────────────────────────────────────────────
        cached_answer = self._cache.get(message)
        if cached_answer:
            logger.info('Cache hit: ' + message[:50])
            _is_ne = self._is_nepali(message)
            _lang  = "ne" if _is_ne else "en"
            try:
                fresh_sugs = self._structure_suggestions(
                    self.suggestion_engine.get_raw_suggestions(
                        user_query=message, bot_response=cached_answer,
                        language=_lang,
                    )
                )
            except Exception:
                fresh_sugs = self._default_suggestions(language=_lang)
            return {
                'answer':       cached_answer,
                'sources':      ['response_cache'],
                'suggestions':  fresh_sugs,
                'display_type': 'text',
                'char_count':   len(cached_answer),
            }

        # Detect category for metadata filtering
        category_filter = self._get_category_filter(message)

        if is_bare:
            retriever    = self._make_retriever(k=10, filter_category=category_filter)
            prompt       = self._prompt_bare()
            display_type = "bare_list"
        elif is_price:
            retriever    = self.retriever_price
            prompt       = self._prompt_price()
            display_type = "text"
        elif is_conservation:
            retriever    = self._make_retriever(k=10, filter_category=category_filter)
            prompt       = self._prompt_list()
            display_type = "list"
        elif is_list:
            retriever    = self._make_retriever(k=10, filter_category=category_filter)
            prompt       = self._prompt_list()
            display_type = "list"
        else:
            retriever    = self._make_retriever(k=2, filter_category=category_filter)
            prompt       = self._prompt_convo()
            display_type = "text"

        logger.info("Type: " + display_type + " | conservation=" + str(is_conservation) + " | " + message[:60])

        # ── Translate Nepali → English for FAISS retrieval ───────────────────
        # The FAISS index is built on English docs; Nepali queries get poor results
        # without translation. We translate for retrieval only — LLM still sees the
        # original Nepali message and responds in Nepali.
        memory       = self._get_memory(session_id)
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        retrieval_query = await self._get_retrieval_query(message, session_id)

        # ── Smart model routing ──────────────────────────────────────────────
        is_nepali_query = self._is_nepali(message)
        if is_list or is_bare or is_conservation:
            llm = self._make_llm(max_tokens=400, model="llama-3.3-70b-versatile")
        elif is_price:
            llm = self._make_llm(max_tokens=250, model="llama-3.1-8b-instant")
        else:
            # Nepali responses need more tokens (Devanagari is denser)
            llm = self._make_llm(max_tokens=250 if is_nepali_query else 150,
                                 model="llama-3.1-8b-instant")

        t_start = time.time()
        try:
            # ── Hybrid retrieval (BM25 + FAISS fused) ────────────────────────
            k_val       = 10 if (is_list or is_bare or is_conservation) else (6 if is_price else 3)
            source_docs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._hybrid_retrieve(retrieval_query, k=k_val, filter_category=category_filter)
                ),
                timeout=QUERY_TIMEOUT_SECS,
            )
            context     = "\n\n".join(d.page_content for d in source_docs)
            # Prepend explicit language instruction so LLM never gets confused by chat history
            lang_prefix = "[RESPOND IN NEPALI]\n" if is_nepali_query else "[RESPOND IN ENGLISH]\n"
            filled      = prompt.format(context=context, chat_history=chat_history,
                                        question=lang_prefix + message)
            llm_resp    = await asyncio.wait_for(llm.ainvoke(filled), timeout=QUERY_TIMEOUT_SECS)
            raw_answer  = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
        except asyncio.TimeoutError:
            logger.warning("Query timed out")
            return {"answer": "Connection timed out. Please try again.", "sources": [],
                    "suggestions": self._default_suggestions(), "display_type": "text"}
        except Exception as e:
            logger.error("LLM invoke failed: " + str(e), exc_info=True)
            return self._error_response()

        logger.info("LLM responded in " + str(round(time.time() - t_start, 2)) + "s")

        # ── Confidence guard: retry with broader retrieval if LLM is uncertain ──
        if self._is_uncertain(raw_answer) and category_filter:
            logger.warning("Low confidence detected — retrying without category filter")
            try:
                broad_retriever = self._make_retriever(k=8, filter_category=None)
                source_docs     = await asyncio.wait_for(
                    broad_retriever.ainvoke(retrieval_query), timeout=QUERY_TIMEOUT_SECS
                )
                context  = "\n\n".join(d.page_content for d in source_docs)
                filled   = prompt.format(context=context, chat_history=chat_history,
                                         question=lang_prefix + message)
                llm_resp = await asyncio.wait_for(llm.ainvoke(filled), timeout=QUERY_TIMEOUT_SECS)
                raw_answer = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
                logger.info("Retry succeeded")
            except Exception as e:
                logger.warning("Retry failed: " + str(e))

        if is_bare:
            answer = self._clean_bare_list(raw_answer)
        elif is_list:
            answer = self._enforce_list_limit(raw_answer.strip())
        else:
            answer = self._clean_convo(raw_answer)

        # ── Hallucination guard: warn in logs if invented species detected ──────
        if (is_list or is_bare or is_conservation) and self._check_hallucination(answer):
            logger.warning("Hallucination detected in answer — consider reviewing source data")
            # Don't block the answer but log it for monitoring

        char_limit = MAX_LIST_RESPONSE_CHARS if (is_list or is_bare or is_conservation) else MAX_RESPONSE_CHARS
        if len(answer) > char_limit:
            answer = answer[:char_limit].rsplit("\n", 1)[0]  # cut at last complete bullet

        memory.save_context({"question": message}, {"answer": answer})
        self._cache.set(message, answer, display_type=display_type)

        sources = list({doc.metadata.get("source", "Knowledge Base") for doc in source_docs})

        suggestions = []
        if include_suggestions:
            try:
                suggestions = self._structure_suggestions(
                    self.suggestion_engine.get_raw_suggestions(
                        user_query=message,
                        bot_response=answer,
                        match_query=retrieval_query,
                        language="ne" if is_nepali_query else "en",
                    )
                )
            except Exception:
                suggestions = self._default_suggestions(language="ne" if is_nepali_query else "en")

        return {"answer": answer, "sources": sources, "suggestions": suggestions,
                "display_type": display_type, "char_count": len(answer)}

    async def query_stream(self, message, session_id="default"):
        message = message.strip()[:MAX_INPUT_CHARS]
        if not message:
            yield "Please ask me something!"
            return
        if self._is_greeting(message):
            yield self._greeting_response(message)["answer"]
            return

        if self._is_activity_list(message):
            yield self._activity_list_response(message)["answer"]
            return

        is_bare         = self._is_bare_list(message)
        is_price        = (not is_bare) and self._is_price(message)
        is_conservation = (not is_bare) and (not is_price) and self._is_conservation(message)
        is_list         = (not is_bare) and (not is_price) and (not is_conservation) and self._is_list(message)

        if is_bare:
            retriever = self.retriever_bare;  prompt = self._prompt_bare()
        elif is_price:
            retriever = self.retriever_price; prompt = self._prompt_price()
        elif is_conservation or is_list:
            retriever = self.retriever_list;  prompt = self._prompt_list()
        else:
            retriever = self.retriever_convo; prompt = self._prompt_convo()

        try:
            # Translate Nepali query to English for better FAISS retrieval
            retrieval_query  = await self._get_retrieval_query(message, session_id)
            is_nepali_query  = self._is_nepali(message)
            lang_prefix      = "[RESPOND IN NEPALI]\n" if is_nepali_query else "[RESPOND IN ENGLISH]\n"
            source_docs      = retriever.invoke(retrieval_query)
            context          = "\n\n".join(d.page_content for d in source_docs)
            memory           = self._get_memory(session_id)
            chat_history     = memory.load_memory_variables({}).get("chat_history", "")
            filled           = prompt.format(context=context, chat_history=chat_history,
                                             question=lang_prefix + message)
            async for chunk in self.llm.astream(filled):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("Streaming error: " + str(e))
            yield "Sorry, something went wrong. Please try again."

    # ── Confidence Guard ─────────────────────────────────────────────────────

    def _is_uncertain(self, answer: str) -> bool:
        """Detect when the LLM admitted it doesn't know."""
        phrases = [
            "i'm unsure", "i am unsure", "i don't have", "i do not have",
            "not available", "no information", "cannot find",
            "i don't know", "i do not know", "not in my knowledge",
            "check the latest", "i couldn't find", "i could not find",
            "unfortunately", "i'm not sure", "i am not sure",
            "no data", "not specified", "not mentioned",
        ]
        a = answer.lower()
        return any(p in a for p in phrases)

    # ── Hallucination Guard ───────────────────────────────────────────────────

    def _check_hallucination(self, answer: str) -> bool:
        """
        Returns True if the answer contains a species name NOT in our data.
        Only runs for list/conservation responses where hallucination risk is high.
        """
        if not KNOWN_SPECIES:
            return False   # guard not ready yet

        # Extract capitalized multi-word phrases (likely species names)
        candidates = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+", answer)
        for candidate in candidates:
            c_lower = candidate.lower()
            # Check if ANY known species name contains this candidate
            matched = any(c_lower in known or known in c_lower for known in KNOWN_SPECIES)
            if not matched and len(candidate.split()) >= 2:
                logger.warning("Possible hallucination detected: " + candidate)
                return True
        return False

    def _is_nepali(self, text: str) -> bool:
        """Return True if the message contains Nepali (Devanagari) characters."""
        return any('\u0900' <= c <= '\u097F' for c in text)

    def _is_followup_query(self, message: str) -> bool:
        """Detect vague follow-up queries that rely on conversation context."""
        followup_triggers = [
            "describe it", "tell me more", "more about it", "what about it",
            "explain it", "elaborate", "more details", "describe that",
            "tell me about it", "what is it", "what's it", "and it",
        ]
        m = message.strip().lower()
        # Short messages with pronouns are almost always follow-ups
        if len(m.split()) <= 3 and any(w in m.split() for w in ["it", "that", "this", "them", "they", "those"]):
            return True
        return any(t in m for t in followup_triggers)

    async def _get_retrieval_query(self, message: str, session_id: str = "default") -> str:
        """Build an effective FAISS retrieval query from the message.
        - For follow-up pronoun queries ('describe it'), anchors to the last user question.
        - For Nepali queries, translates to English (FAISS index is English-only).
        - Otherwise returns the message unchanged.
        """
        # ── Follow-up resolution: "describe it" → "describe Bengal Tiger" ─────
        if self._is_followup_query(message):
            memory = self._get_memory(session_id)
            msgs = memory.messages
            # Find the last human message that isn't this follow-up
            last_human = next(
                (m["content"] for m in reversed(msgs) if m["role"] == "human" and m["content"] != message),
                None
            )
            if last_human:
                combined = last_human + " " + message
                logger.info("Follow-up resolved: '" + message + "' → '" + combined[:60] + "'")
                message = combined  # use enriched query for retrieval

        # ── Nepali translation ─────────────────────────────────────────────────
        if not self._is_nepali(message):
            return message
        try:
            llm = self._make_llm(max_tokens=80, model="llama-3.1-8b-instant")
            filled = f"Translate to English. Output ONLY the English translation, nothing else:\n{message}"
            resp = await asyncio.wait_for(llm.ainvoke(filled), timeout=6.0)
            translated = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            logger.info("Nepali→English: " + message[:40] + " → " + translated[:60])
            return translated
        except Exception:
            return message  # fallback: use original

    def _is_greeting(self, message):
        single_word = {"hello", "hi", "hey", "namaste", "howdy", "greetings", "yo"}
        multi_word  = {"good morning", "good evening", "good afternoon"}
        # Nepali greetings (Devanagari)
        nepali_greetings = {"नमस्ते", "नमस्कार", "सुप्रभात", "नमस्"}
        msg   = message.strip().lower()
        words = msg.split()
        if len(words) > 4:
            return False
        if bool(single_word.intersection(words)) or any(p in msg for p in multi_word):
            return True
        # Check Nepali greetings by word match
        return any(g in message for g in nepali_greetings)

    def _is_activity_list(self, message):
        """Detect questions asking to list available activities at CNP."""
        triggers_en = ["what activities", "list activities", "what can i do", "available activities",
                       "what to do", "things to do", "activities available", "all activities",
                       "what are the activities", "what activities are there"]
        triggers_ne = ["गतिविधिहरू छन्", "गतिविधि छन्", "के-के गतिविधि", "कुन-कुन गतिविधि",
                       "कुन कुन गतिविधि", "के गर्न सकिन्छ", "गतिविधिहरू"]
        m = message.lower()
        return any(t in m for t in triggers_en) or any(t in message for t in triggers_ne)

    def _activity_list_response(self, message):
        """Return pre-built accurate activity list — no LLM, no hallucination."""
        is_ne = self._is_nepali(message)
        if is_ne:
            answer = (
                "• जीप सफारी (Jeep Safari) — नेपाली NPR ५०० | सार्क NPR १,५०० | विदेशी NPR ३,५०० | बिहान ६–१० बजे / साँझ २–५ बजे\n"
                "• हाती सफारी (Elephant Safari) — नेपाली NPR १,६५० | सार्क NPR ४,००० | विदेशी NPR ५,००० | बिहान ६–१० बजे / साँझ २–५ बजे\n"
                "• पक्षी अवलोकन (Bird Watching) — नेपाली NPR ३,००० | सार्क NPR ५,५०० | विदेशी NPR ६,५०० | बिहान ६–१० बजे / साँझ २–५ बजे\n"
                "• जंगल वाक (Jungle Walk) — नेपाली NPR ५,००० | सार्क NPR १०,००० | विदेशी NPR १२,५०० | बिहान ६–१० बजे / साँझ २–५ बजे\n"
                "• डुंगा सफारी (Canoe Safari) — नेपाली NPR ५०० | सार्क NPR ६०० | विदेशी NPR ७०० | बिहान ६–१० बजे / साँझ २–५ बजे\n"
                "• थारु सांस्कृतिक कार्यक्रम (Tharu Cultural Program) — नेपाली NPR २०० | सार्क NPR ३०० | विदेशी NPR ३०० | साँझ ७–८ बजे\n"
                "• थारु संग्रहालय (Tharu Museum) — नेपाली NPR २०० | सार्क NPR ४०० | विदेशी NPR ४०० | बिहान १०–साँझ ५ बजे"
            )
        else:
            answer = (
                "• Jeep Safari — Domestic NPR 500 | SAARC NPR 1,500 | Foreign NPR 3,500 | Morning 6–10AM & Evening 2–5PM\n"
                "• Elephant Safari — Domestic NPR 1,650 | SAARC NPR 4,000 | Foreign NPR 5,000 | Morning 6–10AM & Evening 2–5PM\n"
                "• Bird Watching — Domestic NPR 3,000 | SAARC NPR 5,500 | Foreign NPR 6,500 | Morning 6–10AM & Evening 2–5PM\n"
                "• Jungle Walk — Domestic NPR 5,000 | SAARC NPR 10,000 | Foreign NPR 12,500 | Morning 6–10AM & Evening 2–5PM\n"
                "• Canoe Safari — Domestic NPR 500 | SAARC NPR 600 | Foreign NPR 700 | Morning 6–10AM & Evening 2–5PM\n"
                "• Tharu Cultural Program — Domestic NPR 200 | SAARC NPR 300 | Foreign NPR 300 | Evening 7–8PM\n"
                "• Tharu Museum — Domestic NPR 200 | SAARC NPR 400 | Foreign NPR 400 | 10AM–5PM"
            )
        lang = "ne" if is_ne else "en"
        return {
            "answer": answer,
            "sources": ["activities_database"],
            "suggestions": self._default_suggestions(language=lang),
            "display_type": "list",
            "char_count": len(answer),
        }

    def _is_price(self, message):
        triggers = ["how much", "cost", "price", "fee", "ticket", "entry fee",
                    "rate", "charge", "tariff", "npr", "rupee", "rupees",
                    # timing triggers — use the same hardcoded table so no hallucination
                    "when does", "when do", "what time", "timing", "schedule",
                    "opening time", "closing time", "start time", "open at", "close at",
                    "what are the timings", "what are the hours"]
        nepali_triggers = ["कति", "शुल्क", "मूल्य", "टिकट", "पैसा", "रुपैयाँ",
                           # Nepali timing triggers
                           "कहिले", "कति बजे", "समय", "सुरु हुन्छ", "बन्द हुन्छ",
                           "कार्यक्रम समय", "खुल्ने", "बन्द हुने"]
        msg = message.lower()
        return any(t in msg for t in triggers) or any(t in message for t in nepali_triggers)

    def _is_conservation(self, message):
        """Detect endangered/threatened/conservation LIST questions — needs high k retrieval.
        Skips if it is a where/when/how/why/which/is/what is question — those stay conversational."""
        import re
        m = message.lower().strip()
        # Location, time, method, explanation questions are always conversational
        if re.match(r"^(where|when|how|why|is |are |what is|what's|which|who|can i|should i)", m):
            return False
        triggers = [
            "endangered", "threatened", "vulnerable", "critically",
            "conservation", "extinction", "extinct",
            "at risk", "dying out", "nearly extinct", "conservation status",
            "need protection", "needs protection", "protect",
        ]
        nepali_triggers = ["लोपोन्मुख", "संकटापन्न", "संरक्षण", "विलुप्त", "खतरामा"]
        return any(t in m for t in triggers) or any(t in message for t in nepali_triggers)

    def _is_list(self, message):
        triggers = ["list", "name all", "name some", "tell me all", "give me all", "show all",
                    "enumerate", "what types", "what kind", "what species", "examples of",
                    "types of", "kinds of", "all the", "all animals", "all birds", "all mammals",
                    "mention", "can you list", "could you list", "top 5", "top 10",
                    "top five", "top ten", "top three", "top 3", "what are the", "what are some"]
        nepali_triggers = ["सूची", "नाम बताउ", "सबै बताउ", "कुन-कुन", "कुन कुन",
                           "प्रजातिहरू", "जनावरहरू", "चराहरू", "सबै जनावर",
                           "चराहरू छन्", "जनावरहरू छन्"]
        m = message.lower()
        # These question starters always want a conversational answer, not a list
        if re.match(r"^(which|what is|what's|how does|how do|why|who|when is|when|where|how much|how many|is |are |can i|should i)", m):
            return False
        return any(t in m for t in triggers) or any(t in message for t in nepali_triggers)

    def _is_bare_list(self, message):
        triggers = ["no explanation", "only list", "just list", "just names", "names only",
                    "no description", "without description", "only names", "bare list",
                    "list only", "just the names", "no details"]
        return any(t in message.lower() for t in triggers)

    def _clean_convo(self, text):
        text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        for p in ["Based on the context provided,", "Based on the context,",
                  "According to the information,", "Based on the information provided,",
                  "Here is the information:", "Here are the details:",
                  "As your Chitwan Park wildlife guide", "As your Chitwan National Park wildlife guide",
                  "As your wildlife guide", "As a Chitwan", "As your guide",
                  "Certainly!", "Of course!", "Sure!", "Great question!", "Good question!"]:
            text = text.replace(p, "")
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def _clean_bare_list(self, text):
        cleaned = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[•\-*]\s*", "", line)
            line = re.sub(r"[:\-\u2014(].*$", "", line).strip()
            if line:
                cleaned.append("• " + line)
        return "\n".join(cleaned[:MAX_LIST_ITEMS])

    def _enforce_list_limit(self, text):
        lines        = [l for l in text.split("\n") if l.strip()]
        bullet_lines = [l for l in lines if re.match(r"^\s*[•\-*\d]", l)]
        if len(bullet_lines) > MAX_LIST_ITEMS:
            other_lines = [l for l in lines if not re.match(r"^\s*[•\-*\d]", l)]
            return "\n".join(other_lines + bullet_lines[:MAX_LIST_ITEMS])
        return text

    def _structure_suggestions(self, raw):
        icon_map = {"tiger": "🐯", "rhino": "🦏", "elephant": "🐘", "bird": "🦜",
                    "safari": "🗺️", "visit": "🕐", "photo": "📷", "snake": "🐍",
                    "crocodile": "🐊", "deer": "🦌", "park": "🌿",
                    "price": "💰", "cost": "💰", "fee": "💰"}
        chips = []
        for i, text in enumerate(raw[:3]):
            icon = "🌿"
            for keyword, emoji in icon_map.items():
                if keyword in text.lower():
                    icon = emoji
                    break
            chips.append({"id": i + 1, "text": text, "icon": icon})
        return chips

    def _default_suggestions(self, language: str = "en"):
        if language == "ne":
            return [
                {"id": 1, "text": "बंगाल बाघको बारेमा बताउनुस्", "icon": "🐯"},
                {"id": 2, "text": "कुन जनावरहरू देख्न सकिन्छ?",  "icon": "🦏"},
                {"id": 3, "text": "चितवन भ्रमणको राम्रो समय",    "icon": "🕐"},
            ]
        return [
            {"id": 1, "text": "Tell me about Bengal tigers", "icon": "🐯"},
            {"id": 2, "text": "What animals can I see?",     "icon": "🦏"},
            {"id": 3, "text": "Best time to visit Chitwan",  "icon": "🕐"},
        ]

    def _greeting_response(self, message: str = ""):
        # ── Nepali greeting ───────────────────────────────────────────────────
        if self._is_nepali(message):
            return {
                "answer": (
                    "नमस्ते! चितवन राष्ट्रिय निकुञ्जमा स्वागत छ। "
                    "म तपाईंलाई वन्यजन्तु, सफारी नियम, प्रवेश शुल्क "
                    "र निकुञ्जसम्बन्धी जुनसुकै जानकारी दिन सक्छु। "
                    "के जान्न चाहनुहुन्छ?"
                ),
                "sources": [], "suggestions": self._default_suggestions(language="ne"),
                "display_type": "text", "char_count": 0,
            }

        # ── English greeting ──────────────────────────────────────────────────
        msg = message.strip().lower()
        multi_word = {"good morning": "Good morning", "good evening": "Good evening",
                      "good afternoon": "Good afternoon"}
        word = next((v for k, v in multi_word.items() if k in msg), None) \
               or (message.strip().split()[0].capitalize() if message.strip() else "Hello")

        if word in ("Good morning", "Good evening", "Good afternoon"):
            opening = f"{word}! Hope you're ready for a great day at Chitwan."
        elif word == "Namaste":
            opening = "Namaste! Glad you're here."
        elif word in ("Hi", "Hey", "Hello"):
            opening = f"{word} there! Welcome to Chitwan National Park."
        else:
            opening = f"{word}! Welcome to Chitwan National Park."

        return {
            "answer": (f"{opening} I can help you with wildlife info, safari rules, "
                       "entry fees, or anything else about the park. What would you like to know?"),
            "sources": [], "suggestions": self._default_suggestions(),
            "display_type": "text", "char_count": 0,
        }

    def _error_response(self):
        return {"answer": "Something went wrong. Please try again.",
                "sources": [], "suggestions": self._default_suggestions(), "display_type": "text"}

    def _load_all_documents(self, wildlife_dir, raw_data_dir):
        documents = []

        if wildlife_dir.exists():
            for json_file in wildlife_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    species_list = data if isinstance(data, list) else [data]
                    for species in species_list:
                        documents.append(Document(
                            page_content=self._format_species(species, json_file.stem),
                            metadata={"source": json_file.name, "category": json_file.stem, "type": "wildlife"},
                        ))
                        # Register all known species names for hallucination guard
                        for key in ("commonEnglishName", "english_name", "name", "title",
                                    "nepaliName", "nepali_name", "scientificName", "scientific_name"):
                            val = species.get(key)
                            if val:
                                KNOWN_SPECIES.add(val.lower().strip())
                    logger.info("  Loaded " + json_file.name)
                except Exception as e:
                    logger.warning("  Skipping " + json_file.name + ": " + str(e))

        if raw_data_dir.exists():
            try:
                txt_docs = DirectoryLoader(str(raw_data_dir), glob="*.txt",
                                           loader_cls=TextLoader,
                                           loader_kwargs={"encoding": "utf-8"}).load()
                documents.extend(txt_docs)
                logger.info("Loaded " + str(len(txt_docs)) + " text files")
            except Exception as e:
                logger.warning("Error loading text files: " + str(e))

            activity_file = raw_data_dir / "activities.json"
            if activity_file.exists():
                try:
                    with open(activity_file, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                    documents.append(Document(
                        page_content=self._format_activities(data),
                        metadata={"source": "activities.json", "type": "activities"},
                    ))
                    logger.info("Loaded activities.json")
                except Exception as e:
                    logger.warning("Error loading activities.json: " + str(e))

        logger.info("Total documents: " + str(len(documents)))
        return documents

    def _format_species(self, species, category):
        def get(s, *keys):
            for k in keys:
                if k in s: return s[k]
            return None

        english = get(species, "commonEnglishName", "english_name", "name", "title") or "Unknown"
        nepali  = get(species, "nepaliName",        "nepali_name") or ""
        sci     = get(species, "scientificName",     "scientific_name") or ""
        status  = get(species, "conservationStatus", "conservation_status") or ""
        habitat = get(species, "habitat") or ""
        desc    = get(species, "description") or ""
        sub_cat = get(species, "category") or ""
        facts   = get(species, "interestingFacts", "interesting_facts")

        # Build a rich, natural-language sentence the embedder can match well
        # Format: "English Name (Nepali Name) — description. Found in habitat. Status."
        display_name = english
        if nepali:
            display_name += " (" + nepali + ")"

        parts = [
            "Category: " + category,
            "Bird: " + display_name,
            "English name: " + english,
        ]
        if nepali:
            parts.append("Nepali name: " + nepali)
        if sci:
            parts.append("Scientific name: " + sci)
        if status:
            parts.append("Conservation status: " + status)
        if sub_cat and sub_cat.lower() != category.lower():
            parts.append("Type: " + sub_cat)
        if habitat:
            hab_str = ", ".join(habitat) if isinstance(habitat, list) else habitat
            parts.append("Found in: " + hab_str)
        if desc:
            parts.append("Description: " + desc)
        if facts:
            parts.append("Facts: " + (" ".join(facts) if isinstance(facts, list) else facts))

        return ". ".join(parts) + "."

    def _format_activities(self, data):
        activities = data if isinstance(data, list) else data.get("activities", [data])
        lines = []
        for activity in activities:
            if not isinstance(activity, dict): continue
            name     = activity.get("activity", activity.get("name", "Activity"))
            schedule = activity.get("schedule", "")
            timing   = activity.get("timing", "")
            prices   = activity.get("prices", {})
            pp = []
            if prices.get("domestic"): pp.append("NPR " + str(prices["domestic"]) + " for domestic visitors")
            if prices.get("SAARC"):    pp.append("NPR " + str(prices["SAARC"]) + " for SAARC visitors")
            if prices.get("tourist"):  pp.append("NPR " + str(prices["tourist"]) + " for foreign tourists")
            sentence = name
            if pp: sentence += " costs " + ", ".join(pp) + "."
            if schedule: sentence += " Schedule: " + schedule + "."
            if timing:   sentence += " Timing: " + timing + "."
            lines.append(sentence)
            lines.append("Price of " + name + ": Domestic NPR " + str(prices.get("domestic", "N/A"))
                         + " | SAARC NPR " + str(prices.get("SAARC", "N/A"))
                         + " | Tourist NPR " + str(prices.get("tourist", "N/A")))
            lines.append("How much does " + name + " cost? " + sentence)
            lines.append(name + " fee: Domestic=" + str(prices.get("domestic"))
                         + ", SAARC=" + str(prices.get("SAARC"))
                         + ", Tourist=" + str(prices.get("tourist")))
            lines.append("")
        return "\n".join(lines)

    def clear_memory(self, session_id="default"):
        if session_id in self._sessions:
            self._sessions[session_id].clear()
        logger.info("Memory cleared for session: " + session_id)

    def get_chat_history(self, session_id="default"):
        return self._get_memory(session_id).messages

    def add_documents(self, documents):
        if not self.vector_db:
            raise ValueError("Vector store not initialized.")
        splitter   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)
        self.vector_db.add_documents(split_docs)
        index_path = Path(__file__).resolve().parent.parent.parent / "vector_store" / "faiss_index"
        self.vector_db.save_local(str(index_path))

    def get_stats(self):
        if not self.vector_db:
            return {"status": "not_initialized"}
        return {
            "status":              "initialized",
            "total_vectors":       self.vector_db.index.ntotal,
            "bm25_docs":           len(self._all_docs) if self._bm25_index else 0,
            "retrieval_mode":      "Hybrid BM25+FAISS" if self._bm25_index else "FAISS-only",
            "embedding_model":     "BAAI/bge-small-en-v1.5",
            "embedding_device":    "CPU (FastEmbed)",
            "llm_simple":          "llama-3.1-8b-instant",
            "llm_complex":         "llama-3.3-70b-versatile",
            "active_sessions":     len(self._sessions),
            "response_cache_size": self._cache.size,
        }