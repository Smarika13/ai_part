from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.file_loader import JSONFileLoader
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.vector_store = None
        
    def create_vector_store(self, data_path="data/raw/wildlife"):
        print(f"Loading data from {data_path}...")
        
        # Load JSON documents
        json_loader = JSONFileLoader(data_path)
        documents = json_loader.load_all_json_files()
        
        if not documents:
            raise ValueError(f"No JSON files found in {data_path}")
        
        print(f"Loaded {len(documents)} documents")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"Split into {len(splits)} chunks")
        
        # Create vector store with FAISS
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        # Save to disk
        self.vector_store.save_local("data/processed/faiss_index")
        
        print("✓ Vector store created successfully")
        return self.vector_store
    
    def load_vector_store(self):
        if os.path.exists("data/processed/faiss_index"):
            print("Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                "data/processed/faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Vector store loaded")
        else:
            print("No existing vector store found")
        return self.vector_store