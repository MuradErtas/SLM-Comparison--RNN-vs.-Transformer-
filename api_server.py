"""
FastAPI server for SLM model inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add SLM directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm import load_models, generate_response

# Global model storage
transformer = None
rnn = None
tokenizer = None
executor = ThreadPoolExecutor(max_workers=2)  # For running blocking model inference

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global transformer, rnn, tokenizer
    
    # Startup
    models_path = os.path.join(os.path.dirname(__file__), "models")
    
    if not os.path.exists(models_path):
        raise RuntimeError(f"Models directory not found: {models_path}")
    
    print(f"Loading models from {models_path}...", file=sys.stderr, flush=True)
    start_time = time.time()
    transformer, rnn, tokenizer = load_models(models_path)
    load_time = time.time() - start_time
    print(f"Models loaded successfully in {load_time:.2f}s!", file=sys.stderr, flush=True)
    
    yield
    
    # Shutdown
    print("Shutting down...", file=sys.stderr, flush=True)
    executor.shutdown(wait=False)

app = FastAPI(title="SLM API", version="1.0.0", lifespan=lifespan)

# CORS configuration - allow Next.js frontend
# Note: FastAPI doesn't support wildcards like "https://*.vercel.app"
# Use "*" for all origins or list specific domains
allowed_origins = os.getenv("ALLOWED_ORIGIN", "*")
if allowed_origins != "*":
    # If specific origin provided, use it (comma-separated for multiple)
    allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]
    allowed_origins.append("http://localhost:3000")  # Always allow local dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if isinstance(allowed_origins, list) else ["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

@app.get("/")
async def root():
    """Health check endpoint - should respond immediately."""
    return {
        "status": "ok",
        "models_loaded": transformer is not None and rnn is not None,
        "service": "SLM API"
    }

@app.get("/health")
async def health():
    """Health check with model status - should respond immediately."""
    return {
        "status": "healthy",
        "models_loaded": transformer is not None and rnn is not None,
        "service": "SLM API"
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate responses from both Transformer and RNN models."""
    if not transformer or not rnn or not tokenizer:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    start_time = time.time()
    print(f"[{time.time():.2f}] Received request: '{request.prompt[:50]}...' (max_tokens={request.max_tokens})", file=sys.stderr, flush=True)
    
    try:
        # Run blocking model inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        transformer_text, rnn_text = await loop.run_in_executor(
            executor,
            generate_response,
            request.prompt,
            transformer,
            rnn,
            tokenizer,
            request.max_tokens
        )
        
        duration = time.time() - start_time
        print(f"[{time.time():.2f}] Generation complete in {duration:.2f}s", file=sys.stderr, flush=True)
        
        return {
            "transformer": transformer_text,
            "rnn": rnn_text
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"[{time.time():.2f}] Error after {duration:.2f}s: {e}", file=sys.stderr, flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
