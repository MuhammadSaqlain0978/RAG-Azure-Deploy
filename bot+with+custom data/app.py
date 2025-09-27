from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import PyPDF2
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from mistralai import Mistral
import docx2txt
from docx import Document as DocxDocument

app = FastAPI(title="University RAG API", version="1.0.0")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    answer: str
    sources: list
    context_used: int
    source_types: list
    status: str

# Initialize bot instance
bot = None

def initialize_rag_bot():
    global bot
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set")
        
        # Import your UniversityRAGBot class here
        # For now, using a simplified version
        bot = SimpleRAGBot(api_key)
        print("✅ RAG Bot initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG bot: {e}")

class SimpleRAGBot:
    def __init__(self, api_key):
        self.mistral_client = Mistral(api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # In production, you'd load your vector store here
        self.vector_store = None

    def answer_query(self, question):
        # Simplified version for demo
        prompt = f"""You are a helpful university assistant. Answer this question helpfully and accurately.

Question: {question}

Answer:"""
        
        response = self.mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': [],
            'context_used': 0,
            'source_types': []
        }

# Initialize on startup
initialize_rag_bot()

@app.get("/")
async def root():
    return {"message": "University RAG API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "bot_initialized": bot is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if bot is None:
        raise HTTPException(status_code=503, detail="RAG bot not initialized")
    
    try:
        result = bot.answer_query(request.message)
        return ChatResponse(**result, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)