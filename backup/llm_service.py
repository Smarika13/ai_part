from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    def __init__(self, vector_store):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
            convert_system_message_to_human=True,
            max_output_tokens=150,  # Slightly increased from 100
            top_p=0.85
        )

        self.vector_store = vector_store
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        template = """You're a friendly Chitwan park guide. Answer briefly and naturally.

Context: {context}

Question: {question}

Rules:
- Keep it short (2-3 sentences max)
- Sound conversational like texting a friend
- Only give detailed answers if asked
- No bullet points unless listing multiple things
- No "based on context" phrases
- Start with the answer directly

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 2}  # Changed from 1 to 2 for better context
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def normalize_query(self, query: str) -> str:
        """Normalize casual language to improve retrieval"""
        replacements = {
            " u ": " you ", 
            " r ": " are ", 
            " pls ": " please ",
            " plz ": " please ", 
            " info ": " information ",
            " w/ ": " with ", 
            " w/o ": " without ", 
            " thx ": " thanks ",
            " thnx ": " thanks ",
            " ur ": " your ", 
            " abt ": " about ", 
            " n ": " and ",
        }
        
        normalized = " " + query.lower() + " "
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized.strip()

    def get_response(self, query: str):
        """Get response and clean up robotic phrases"""
        normalized_query = self.normalize_query(query)
        response = self.qa_chain.invoke({"query": normalized_query})
        
        # Clean up the answer
        answer = response["result"]
        answer = answer.replace("Based on the provided context, ", "")
        answer = answer.replace("Based on the context, ", "")
        answer = answer.replace("According to the context, ", "")
        answer = answer.replace("Based on the information provided, ", "")

        return {
            "answer": answer,
            "sources": [doc.metadata for doc in response.get("source_documents", [])]
        }