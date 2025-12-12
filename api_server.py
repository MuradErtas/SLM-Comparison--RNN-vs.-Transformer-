"""
FastAPI server for SLM model inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os

# Add SLM directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm import load_models, generate_response

# Global model storage
transformer = None
rnn = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global transformer, rnn, tokenizer
    
    # Startup
    models_path = os.path.join(os.path.dirname(__file__), "models")
    
    if not os.path.exists(models_path):
        raise RuntimeError(f"Models directory not found: {models_path}")
    
    print(f"Loading models from {models_path}...", file=sys.stderr)
    transformer, rnn, tokenizer = load_models(models_path)
    print("Models loaded successfully!", file=sys.stderr)
    
    yield
    
    # Shutdown (cleanup if needed)
    # Models will be cleaned up automatically when process ends

app = FastAPI(title="SLM API", version="1.0.0", lifespan=lifespan)

# CORS configuration - allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        os.getenv("ALLOWED_ORIGIN", "*"),  # Set in production
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": transformer is not None and rnn is not None
    }

@app.get("/health")
async def health():
    """Health check with model status."""
    return {
        "status": "healthy",
        "models_loaded": transformer is not None and rnn is not None
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate responses from both Transformer and RNN models."""
    if not transformer or not rnn or not tokenizer:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        transformer_text, rnn_text = generate_response(
            request.prompt,
            transformer,
            rnn,
            tokenizer,
            max_tokens=request.max_tokens
        )
        
        return {
            "transformer": transformer_text,
            "rnn": rnn_text
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
