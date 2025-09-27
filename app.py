from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ✅ IMPORT YOUR ACTUAL RAG BOT
# Replace "your_bot_file" with the actual filename of your bot code
from custom_data_bot import UniversityRAGBot  # Change this to your actual filename

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
        
        # ✅ USE YOUR ACTUAL RAG BOT
        bot = UniversityRAGBot(api_key=api_key, model_name="mistral-small-latest")
        
        # ✅ MODIFIED FOR AZURE - Don't load local files on Azure
        # On Azure, we'll use pre-built vector store or different approach
        print("🤖 RAG Bot initialized (Azure mode - no local data loading)")
        
        # For now, we'll use the bot without local data loading
        # You'll need to implement Azure Storage loading later
        print("✅ RAG Bot initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing RAG bot: {e}")
        raise e

# Initialize on startup
@app.on_event("startup")
async def startup_event():
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
        # For now, use a simple response since we don't have vector store on Azure yet
        # This will at least confirm the API is working
        test_response = {
            'answer': 'Hello! The RAG API is working. Vector store loading needs to be implemented for Azure.',
            'sources': [],
            'context_used': 0,
            'source_types': []
        }
        return ChatResponse(**test_response, status="success")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)