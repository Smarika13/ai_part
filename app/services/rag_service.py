"""
RAG Service for Chitwan National Park Chatbot
Handles document loading, embedding, and retrieval
"""

import os
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
print(f"DEBUG: Looking for .env at: {env_path}")
print(f"DEBUG: .env exists: {env_path.exists()}")
load_dotenv(dotenv_path=env_path)



class RAGService:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.data_dir = Path(__file__).parent.parent / "data" / "raw"
        self.persist_dir = Path(__file__).parent.parent / "vector_store" / "chroma_db"
        
    def initialize(self, rebuild_index=False):
        """Initialize the RAG service and load/create vector store"""
        print("\n🔧 Initializing RAG Service...")
        
        # Initialize embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        print(f"DEBUG: API key loaded: {api_key_check is not None}")
        if api_key_check:
            print(f"DEBUG: API key starts with: {api_key_check[:10]}...")


        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        print("✓ Embeddings model loaded")
        
        # Load or create vector store
        if rebuild_index or not self.persist_dir.exists():
            print("\n📚 Building new vector index...")
            self._build_index()
        else:
            print("\n📂 Loading existing vector index...")
            self._load_index()
            
    def _build_index(self):
        """Build vector index from data directory"""
        documents = self._load_documents()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_dir}")
        
        print(f"✓ Loaded {len(documents)} documents")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(chunks)} chunks")
        
        # Create vector store
        print("🔄 Creating embeddings and storing in ChromaDB...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        print("✓ Vector store created and persisted")
        
    def _load_index(self):
        """Load existing vector index"""
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings
        )
        print("✓ Vector store loaded")
        
    def _load_documents(self):
        """Load documents from data directory"""
        documents = []
        
        # Load text files
        txt_files = list(self.data_dir.rglob("*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file))
                documents.extend(loader.load())
                print(f"✓ Loaded {txt_file.name}")
            except Exception as e:
                print(f"⚠ Warning loading {txt_file.name}: {e}")
        
        # Load JSON files
        json_files = list(self.data_dir.rglob("*.json"))
        for json_file in json_files:
            try:
                # Basic JSON loader - adjust jq_schema based on your JSON structure
                loader = JSONLoader(
                    file_path=str(json_file),
                    jq_schema='.[]',
                    text_content=False
                )
                documents.extend(loader.load())
                print(f"✓ Loaded {json_file.name}")
            except Exception as e:
                print(f"⚠ Warning loading {json_file.name}: {e}")
        
        return documents
    
    def get_retriever(self, k=4):
        """Get retriever for querying the vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def search(self, query, k=4):
        """Search for relevant documents"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        return self.vectorstore.similarity_search(query, k=k)