import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI

from ingest import run_ingestion, is_index_available
from rag import query as rag_query, get_metadata, is_loaded as is_rag_loaded
from audio_utils import save_audio_file, cleanup_audio_file, validate_audio_file


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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return OpenAI(api_key=api_key)


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
    audio_data = await audio.read()
    
    is_valid, message = validate_audio_file(audio_data, audio.filename or "audio.webm")
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    use_whisper_api = os.environ.get("WHISPER_API", "true").lower() == "true"
    
    if use_whisper_api:
        try:
            client = get_openai_client()
            
            audio_path = save_audio_file(audio_data, audio.filename or "audio.webm")
            
            try:
                with open(audio_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json"
                    )
                
                return TranscriptionResponse(
                    transcript=response.text,
                    language=getattr(response, 'language', 'auto-detected')
                )
            finally:
                cleanup_audio_file(audio_path)
                
        except Exception as e:
            print(f"Whisper API error: {e}")
            return TranscriptionResponse(
                transcript="[Demo mode] I am a farmer and my crops were damaged by heavy rain. I need help.",
                language="en"
            )
    else:
        return TranscriptionResponse(
            transcript="[Demo mode - Whisper disabled] Sample transcription for testing.",
            language="en"
        )


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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
