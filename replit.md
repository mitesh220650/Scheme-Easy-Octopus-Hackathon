# Scheme-Easy Project

## Overview
Scheme-Easy is a multilingual voice-first chatbot application that helps users discover government welfare schemes they qualify for. It uses a RAG (Retrieval-Augmented Generation) pipeline with FAISS vector search and OpenAI for intelligent scheme matching.

## Project Structure
```
scheme-easy/
├── backend/
│   ├── app.py           # FastAPI server with all endpoints
│   ├── ingest.py        # Document ingestion and embedding pipeline
│   ├── rag.py           # RAG retrieval and LLM generation
│   ├── audio_utils.py   # Audio file handling utilities
│   ├── models/          # FAISS index and metadata storage
│   └── sample_data/     # 6 sample government scheme documents
├── frontend-react/
│   ├── src/
│   │   ├── App.jsx      # Main React component with voice UI
│   │   ├── main.jsx     # React entry point
│   │   └── index.css    # Tailwind CSS styles
│   ├── vite.config.js   # Vite configuration with API proxy
│   └── package.json     # Frontend dependencies
├── .env.template        # Environment variable template
└── README.md            # Project documentation
```

## Architecture
- **Backend (Port 8000)**: FastAPI server handling transcription, RAG queries, and document ingestion
- **Frontend (Port 5000)**: React app with voice recording, transcription display, and accessible UI
- **Vector Store**: FAISS for local similarity search
- **LLM**: OpenAI GPT for response generation
- **STT**: OpenAI Whisper for speech-to-text

## Key Endpoints
- `GET /health` - Health check
- `POST /transcribe` - Audio to text
- `POST /query` - Text to scheme recommendations
- `POST /ask-audio` - Combined audio query
- `GET /metadata` - List schemes
- `POST /ingest` - Rebuild index

## Configuration
Required environment variable:
- `OPENAI_API_KEY` - OpenAI API key for embeddings, LLM, and Whisper

Optional:
- `EMBEDDING_MODEL` - Default: text-embedding-3-small
- `LLM_MODEL` - Default: gpt-4o-mini
- `WHISPER_API` - Default: true

## Recent Changes
- Initial project setup with full RAG pipeline
- 6 sample Indian government schemes included
- React frontend with voice recording and TTS playback
- FAISS vector search for scheme matching
