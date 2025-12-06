# --- 1. Load Environment Variables First ---
from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

# --- 2. Import Whisper for Local Transcription ---
import whisper

from ingest import run_ingestion, is_index_available
from rag import query as rag_query, get_metadata, is_loaded as is_rag_loaded
from audio_utils import save_audio_file, cleanup_audio_file, validate_audio_file

# --- 3. Load Local Whisper Model (Global Variable) ---
# "base" is a good balance of speed and accuracy. 
# Options: tiny, base, small, medium, large
print("Loading local Whisper model...")
whisper_model = whisper.load_model("medium")
print("Whisper model loaded.")


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    language: str = "auto"


class TranscriptionResponse(BaseModel):
    transcript: str
    language: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    schemes_count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not is_index_available():
        print("Index not found, running ingestion on startup...")
        try:
            result = run_ingestion()
            print(f"Ingestion result: {result}")
        except Exception as e:
            print(f"Warning: Could not run ingestion on startup: {e}")
            print("You may need to set OPENAI_API_KEY and call /ingest manually")
    else:
        print("Index already exists, loading...")
        is_rag_loaded()
    
    yield


app = FastAPI(
    title="Scheme-Easy API",
    description="Multilingual voice-first chatbot for government welfare schemes",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_client() -> OpenAI:
    # Use local Ollama settings
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    loaded = is_rag_loaded()
    metadata = get_metadata() if loaded else []
    
    return HealthResponse(
        status="healthy",
        index_loaded=loaded,
        schemes_count=len(metadata)
    )


@app.post("/ingest")
async def ingest_documents():
    try:
        result = run_ingestion()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    # 1. Read the audio file
    audio_data = await audio.read()
    
    # 2. Validate file (size, format, etc.)
    is_valid, message = validate_audio_file(audio_data, audio.filename or "audio.webm")
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # 3. Save temporarily for Whisper to process
    # We use a temporary file because Whisper requires a file path
    temp_filename = f"temp_{audio.filename}"
    
    try:
        with open(temp_filename, "wb") as f:
            f.write(audio_data)
        
        # 4. Transcribe using local Whisper model
        # This runs on CPU (or GPU if available/configured)
        result = whisper_model.transcribe(temp_filename)
        text = result["text"].strip()
        
        # Whisper auto-detects language, but the 'result' object has more details
        # For simplicity, we default to 'en' or rely on what's detected
        # You can access result['language'] if needed
        detected_lang = result.get('language', 'en')

        return TranscriptionResponse(
            transcript=text,
            language=detected_lang
        )

    except Exception as e:
        print(f"Transcription error: {e}")
        return TranscriptionResponse(
            transcript="Error: Could not transcribe audio.",
            language="en"
        )
        
    finally:
        # 5. Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/query")
async def query_schemes(request: QueryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    if not is_rag_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Please run /ingest first."
        )
    
    try:
        result = rag_query(
            text=request.text,
            top_k=request.top_k,
            language=request.language
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/ask-audio")
async def ask_with_audio(
    audio: UploadFile = File(...),
    top_k: int = Form(default=5),
    language: str = Form(default="auto")
):
    # This now calls our updated local transcribe function
    transcription = await transcribe_audio(audio)
    
    if not is_rag_loaded():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Please run /ingest first."
        )
    
    try:
        result = rag_query(
            text=transcription.transcript,
            top_k=top_k,
            language=language
        )
        
        result["transcription"] = {
            "text": transcription.transcript,
            "language": transcription.language
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/metadata")
async def get_schemes_metadata():
    if not is_rag_loaded():
        return {"schemes": [], "message": "Index not loaded. Run /ingest first."}
    
    metadata = get_metadata()
    return {"schemes": metadata, "count": len(metadata)}


if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 for local, keep 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)