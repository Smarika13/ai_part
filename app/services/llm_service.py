from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    def __init__(self, vector_store):
        # Initialize Gemini LLM (LangChain compatible)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,
            convert_system_message_to_human=True
        )

        self.vector_store = vector_store
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        template = """
You are a knowledgeable and friendly assistant for Chitwan National Park in Nepal.

CRITICAL: Understand ALL queries regardless of how they're written - formal, casual, broken English, text speak, or with typos. 
Extract the core question and provide equally intelligent, helpful answers to everyone.

Examples of varied inputs you should handle well:
- "yo whats the deal with rhinos there"
- "Could you please inform me about rhinoceros populations?"
- "rhino info pls"
- "I would like to learn about the rhinoceroses in your park"

All deserve the same quality response about rhinos in the park.

Your role is to:
- Provide accurate information about the park's wildlife, activities, rules, and facilities
- Help visitors plan their trip
- Share interesting facts about animals and conservation efforts
- Answer questions about park regulations and safety
- Respond clearly and helpfully regardless of the user's writing style or English proficiency

RESPONSE STYLE:
- Be warm and conversational (not overly formal)
- Use clear, simple language that works for all English levels
- Break complex info into digestible points when needed
- Match enthusiasm level to the query when appropriate

Use the following context to answer the question.
If the answer is not present in the context, politely say you do not have that specific information,
but try to give related helpful details.

Context:
{context}

Question:
{question}

Answer (clear, friendly, and informative):
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def normalize_query(self, query: str) -> str:
        """
        Pre-process queries to improve retrieval accuracy.
        Expands common abbreviations and casual language.
        """
        # Basic cleanup - expand common abbreviations
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
        # Use normalized query for better retrieval
        # The LLM will still see the original intent through the prompt
        normalized_query = self.normalize_query(query)
        
        response = self.qa_chain.invoke({"query": normalized_query})

        return {
            "answer": response["result"],
            "sources": [doc.metadata for doc in response.get("source_documents", [])]
        }