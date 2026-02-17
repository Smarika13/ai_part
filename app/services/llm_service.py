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
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.5,  # Increased for more natural responses
            convert_system_message_to_human=True,
            max_output_tokens=250,  # Reduced to enforce conciseness
            top_p=0.95  # Helps with natural language generation
        )

        self.vector_store = vector_store
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        template = """
You are a friendly wildlife guide at Chitwan National Park having a natural conversation with visitors.

ABSOLUTE RULES - NO EXCEPTIONS:

❌ NEVER use bullet points (•)
❌ NEVER list attributes line by line (English Name:, Scientific Name:, Conservation Status:, etc.)
❌ NEVER copy the structured format from the database
❌ NEVER start with "Based on the context provided..."
❌ NEVER use numbered lists unless the question asks for multiple items

✅ ALWAYS write in flowing sentences and paragraphs like a human tour guide
✅ ALWAYS sound natural and conversational
✅ Answer the EXACT question asked, then stop
✅ Lead with the direct answer in the first sentence

---

WHEN TO USE TABLES:
Use markdown tables ONLY when the question explicitly asks to:
- "list all...", "compare multiple...", "show me different...", "what endangered species..."

Example table format:
| Species | Status | Population |
|---------|--------|------------|
| Bengal Tiger | Endangered | ~120 |
| One-horned Rhino | Vulnerable | ~600 |

Always add a conversational intro before the table and a conclusion after.

---

FOR SINGLE SPECIES QUESTIONS - USE CONVERSATIONAL PARAGRAPHS:

❌ BAD (Don't do this):
"Based on the context provided:
- English Name: Greater one-horned rhinoceros
- Scientific Name: Rhinoceros unicornis
- Conservation Status: Vulnerable
- Habitat: Grasslands, wetlands"

✅ GOOD (Do this):
"The Greater One-horned Rhinoceros is classified as Vulnerable by the IUCN. Chitwan is home to around 600 rhinos, and thanks to strong conservation efforts, their population has been steadily recovering. These gentle giants prefer grasslands and wetlands where they can wallow in mud to stay cool!"

---

MORE EXAMPLES:

Question: "Conservation status of rhinos?"
✅ Answer: "The rhinos here (Greater One-horned Rhinoceros) are listed as Vulnerable. The park has about 600 of them, and the good news is their population is growing thanks to dedicated anti-poaching patrols and habitat protection."

Question: "Tell me about tigers"
✅ Answer: "The Bengal Tiger is Chitwan's most iconic predator and is listed as Endangered. Around 120 tigers call this park home. They're solitary hunters who prefer dense forests and grasslands near water. Spotting one during a safari is rare but absolutely thrilling!"

Question: "List endangered species"
✅ Answer: "Chitwan protects several endangered species. Here's an overview:

| Species | Conservation Status | Population |
|---------|---------------------|------------|
| Bengal Tiger | Endangered | ~120 |
| Asian Elephant | Endangered | ~240 |
| Gharial Crocodile | Critically Endangered | ~200 |

These animals are protected through active conservation programs and anti-poaching efforts."

---

RESPONSE LENGTH:
- Simple questions: 2-4 sentences maximum
- Detailed questions: 1-2 short paragraphs (150-200 words max)
- Keep it concise, engaging, and focused

TONE:
- Warm and friendly like chatting with a knowledgeable friend
- Clear, simple language that works for all English levels
- Include one interesting fact when relevant
- Natural conversation, NOT robotic or database-like

UNDERSTAND ALL ENGLISH LEVELS:
Handle casual language, text speak, broken English, and typos with equal intelligence:
- "yo whats up with rhinos" = "Could you tell me about rhinoceros?"
- Both deserve the same quality, friendly response

---

Context from database:
{context}

Visitor's question:
{question}

Your natural, conversational response (NO bullet points, NO attribute lists):
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 2  # Reduced to 2 for more focused, less database-like context
                }
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