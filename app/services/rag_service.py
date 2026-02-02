import os
import json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, HumanMessage, AIMessage  # ← ADDED: Import message types
from langchain.prompts import PromptTemplate


# Import suggestion engine and emoji formatter
import sys
sys.path.append(str(Path(__file__).parent))
from suggestion_engine import SuggestionEngine
from emoji_formatter import EmojiFormatter


class RAGService:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self.vector_db = None
        self.qa_chain = None
        self.memory = None  # ← ADDED: Conversation memory
        self.suggestion_engine = SuggestionEngine()  # ← NEW: Suggestion engine
        self.emoji_formatter = EmojiFormatter()  # ← NEW: Emoji formatter


    def initialize(self, rebuild_index=False):
        """
        Initialize the RAG service with FAISS vector store and conversation memory
        Uses LOCAL embeddings (no API quota limits!)
        Args:
            rebuild_index: If True, rebuilds the FAISS index from scratch
        """
        print("🔧 Initializing RAG Service with FAISS and Conversation Memory...")
        
        # 1. Get API Key (only for LLM, not embeddings)
        api_key = os.getenv("GOOGLE_API_KEY")
        
        print(f"🔍 Checking for API key...")
        
        if not api_key:
            print("❌ ERROR: GOOGLE_API_KEY not found in environment variables")
            raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")
        
        print(f"✅ API key loaded (starts with: {api_key[:10]}...)")


        # 2. Initialize LOCAL embeddings (runs on your computer, no API calls!)
        print("🤖 Initializing LOCAL embeddings (sentence-transformers)...")
        print("   This runs on your computer - no API quota limits!")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Local embeddings initialized successfully (no quota limits!)")
        except Exception as e:
            print(f"❌ Error initializing embeddings: {e}")
            raise


        # 3. Initialize Google Gemini LLM (only for chat responses)
        print("🤖 Initializing Google Gemini LLM for chat responses...")
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-flash-latest",
                temperature=0.1,
                google_api_key=api_key
            )
            print("✅ Gemini LLM initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing LLM: {e}")
            raise


        # 4. Setup paths
        base_dir = Path(__file__).resolve().parent.parent.parent
        wildlife_dir = base_dir / "wildlife"
        raw_data_dir = base_dir / "app" / "data" / "raw"
        vector_store_dir = base_dir / "vector_store"
        index_path = vector_store_dir / "faiss_index"


        print(f"📂 Base directory: {base_dir}")
        print(f"📂 Wildlife directory: {wildlife_dir}")
        print(f"📂 Raw data directory: {raw_data_dir}")
        print(f"📂 FAISS index path: {index_path}")


        # 5. Load or Build FAISS Index
        if not rebuild_index and index_path.exists():
            print("📁 Loading existing FAISS index...")
            try:
                self.vector_db = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded FAISS index with {self.vector_db.index.ntotal} vectors")
            except Exception as e:
                print(f"⚠️ Error loading index: {e}")
                print("🔄 Rebuilding index from scratch...")
                rebuild_index = True


        if rebuild_index or not self.vector_db:
            print("🏗️ Building new FAISS index from source data...")
            documents = self._load_all_documents(wildlife_dir, raw_data_dir)
            
            if not documents:
                raise ValueError("⚠️ No documents found! Check your wildlife and raw data folders.")


            # Split documents into chunks
            print(f"📄 Splitting {len(documents)} documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            final_docs = splitter.split_documents(documents)
            print(f"✂️ Created {len(final_docs)} document chunks")
            
            # Create FAISS index (using LOCAL embeddings - no API calls!)
            print("🔨 Creating FAISS vector store with local embeddings...")
            print("   (This happens on your computer, no quota limits!)")
            self.vector_db = FAISS.from_documents(final_docs, self.embeddings)
            
            # Save the index
            vector_store_dir.mkdir(parents=True, exist_ok=True)
            self.vector_db.save_local(str(index_path))
            print(f"💾 FAISS index saved to {index_path}")
            print(f"✅ Index contains {self.vector_db.index.ntotal} vectors")


        # 6. Initialize Conversation Memory ← NEW!
        print("🧠 Initializing conversation memory...")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        print("✅ Conversation memory initialized")


        # 7. Build the Conversational QA Chain ← UPDATED!
        print("🔗 Building Conversational QA Chain...")
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=False,
            # You can add a custom combine_docs_chain_kwargs if needed
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template="""Use the following context to answer the question. The context contains information about wildlife in Chitwan National Park including birds, mammals, reptiles, and park information.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer the question based on the context above. Be specific and include relevant details like scientific names, Nepali names, conservation status, and habitat information when available. If you're listing species, provide clear information about each one.

Answer:""",
                    input_variables=["context", "chat_history", "question"]
                )
            }
        )
        print("✅ RAG Service Initialized Successfully with Conversation Memory!")


    def _load_all_documents(self, wildlife_dir: Path, raw_data_dir: Path) -> list:
        """Load all documents from wildlife JSONs and raw data files"""
        documents = []


        # A. Process JSON files in the 'wildlife' folder
        if wildlife_dir.exists():
            print(f"📚 Loading wildlife JSONs from {wildlife_dir}...")
            json_count = 0
            for json_file in wildlife_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8-sig") as f:  # ← UPDATED encoding
                        data = json.load(f)
                        species_list = data if isinstance(data, list) else [data]
                        
                        for species in species_list:
                            # Create searchable content from JSON
                            content = f"Category: {json_file.stem}\n{json.dumps(species, indent=2)}"
                            documents.append(Document(
                                page_content=content,
                                metadata={
                                    "source": json_file.name,
                                    "category": json_file.stem,
                                    "type": "wildlife"
                                }
                            ))
                    json_count += 1
                    print(f"   ✅ Loaded {json_file.name}")
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ Skipping {json_file.name}: Invalid JSON - {e}")
                except Exception as e:
                    print(f"   ⚠️ Error reading {json_file.name}: {e}")
            print(f"✅ Successfully loaded {json_count} wildlife JSON files")
        else:
            print(f"⚠️ Wildlife directory not found: {wildlife_dir}")


        # B. Process text files in app/data/raw
        if raw_data_dir.exists():
            print(f"📚 Loading text files from {raw_data_dir}...")
            try:
                loader = DirectoryLoader(
                    str(raw_data_dir),
                    glob="*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                txt_docs = loader.load()
                documents.extend(txt_docs)
                print(f"✅ Loaded {len(txt_docs)} text files")
            except Exception as e:
                print(f"⚠️ Error loading text files: {e}")
            
            # C. Load activities.json if it exists
            activity_file = raw_data_dir / "activities.json"
            if activity_file.exists():
                try:
                    with open(activity_file, "r", encoding="utf-8-sig") as f:  # ← UPDATED encoding
                        data = json.load(f)
                        # Only add if file is not empty
                        if data:
                            documents.append(Document(
                                page_content=json.dumps(data, indent=2),
                                metadata={
                                    "source": "activities.json",
                                    "type": "activities"
                                }
                            ))
                            print("✅ Loaded activities.json")
                        else:
                            print("⚠️ activities.json is empty, skipping")
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping activities.json: Invalid JSON - {e}")
                except Exception as e:
                    print(f"⚠️ Error loading activities.json: {e}")
        else:
            print(f"⚠️ Raw data directory not found: {raw_data_dir}")


        print(f"\n📊 Total documents loaded: {len(documents)}")
        return documents


    def query(self, message: str, include_suggestions: bool = True, use_emojis: bool = True) -> dict:
        """
        Query the RAG system with conversation memory, smart suggestions, and emoji formatting
        Args:
            message: User query string
            include_suggestions: Whether to include follow-up suggestions
            use_emojis: Whether to format response with emojis
        Returns:
            dict with 'answer', 'sources', and 'suggestions' keys
        """
        if not self.qa_chain:
            return {
                "answer": "RAG system is not initialized. Please call initialize() first.",
                "sources": [],
                "suggestions": []
            }
        
        try:
            # ✨ CRITICAL FIX: Clear the memory before this query to prevent double-adding
            # The qa_chain will add messages automatically, so we need to intercept
            
            # Save current memory state
            current_history = self.memory.chat_memory.messages.copy()
            
            # Use 'question' key for ConversationalRetrievalChain
            result = self.qa_chain.invoke({"question": message})
            
            # Get the raw answer
            raw_answer = result["answer"]
            
            # Format answer with emojis if enabled
            formatted_answer = self.emoji_formatter.format_response(raw_answer) if use_emojis else raw_answer
            
            # Extract unique sources
            sources = list(set([
                doc.metadata.get("source", "Knowledge Base")
                for doc in result.get("source_documents", [])
            ]))
            
            # Generate smart suggestions
            suggestions = []
            if include_suggestions:
                suggestions = self.suggestion_engine.get_suggestions(
                    user_query=message,
                    bot_response=raw_answer
                )
            
            # ✨ NEW: Build the complete assistant message with suggestions
            complete_answer = formatted_answer
            
            if suggestions:
                complete_answer += "\n\n💡 **You might also want to know:**"
                for i, suggestion in enumerate(suggestions, 1):
                    complete_answer += f"\n{i}. {suggestion}"
            
            # ✨ CRITICAL FIX: Update the last assistant message in memory with suggestions
            # The qa_chain already added the user message and assistant response
            # We need to replace the assistant's message with our complete version
            
            if len(self.memory.chat_memory.messages) >= 2:
                # Remove the last assistant message that was auto-added
                self.memory.chat_memory.messages.pop()
                
                # Add our complete message with suggestions
                self.memory.chat_memory.add_message(AIMessage(content=complete_answer))
            
            return {
                "answer": complete_answer,  # ← Return the complete answer with suggestions
                "sources": sources,
                "suggestions": suggestions  # ← Still return suggestions separately for UI
            }
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "suggestions": []
            }


    def clear_memory(self):
        """Clear conversation history - start a new conversation"""
        if self.memory:
            self.memory.clear()
            print("🧹 Conversation memory cleared - starting fresh!")
        else:
            print("⚠️ Memory not initialized")


    def get_chat_history(self) -> list:
        """Get current chat history"""
        if self.memory:
            history = self.memory.load_memory_variables({}).get("chat_history", [])
            return history
        return []


    def add_documents(self, documents: list):
        """
        Add new documents to the existing FAISS index
        Args:
            documents: List of Document objects to add
        """
        if not self.vector_db:
            raise ValueError("Vector store not initialized")
        
        print(f"➕ Adding {len(documents)} documents to FAISS index...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)
        
        self.vector_db.add_documents(split_docs)
        
        # Save updated index
        base_dir = Path(__file__).resolve().parent.parent.parent
        index_path = base_dir / "vector_store" / "faiss_index"
        self.vector_db.save_local(str(index_path))
        print(f"✅ Added documents. Index now contains {self.vector_db.index.ntotal} vectors")


    def get_stats(self) -> dict:
        """Get statistics about the FAISS index and conversation"""
        if not self.vector_db:
            return {"status": "not_initialized"}
        
        chat_history = self.get_chat_history()
        
        return {
            "status": "initialized",
            "total_vectors": self.vector_db.index.ntotal,
            "embedding_dimension": self.vector_db.index.d,
            "embedding_model": "all-MiniLM-L6-v2 (local)",
            "conversation_turns": len(chat_history),
            "memory_enabled": self.memory is not None
        }