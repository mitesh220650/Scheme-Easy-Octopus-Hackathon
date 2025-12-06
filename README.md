# Scheme-Easy / Benefit-Bridge

A multilingual, voice-first chatbot that helps users find government welfare schemes they qualify for using a RAG (Retrieval-Augmented Generation) pipeline.

## Features

- **Voice Input**: Speak your situation in your native language
- **Multilingual Support**: Auto-detects language and responds accordingly
- **RAG Pipeline**: Uses FAISS vector search + OpenAI embeddings for accurate scheme matching
- **Simple Language**: Responses use easy-to-understand words for all users
- **Document Checklist**: Get exact documents needed for each scheme
- **Text-to-Speech**: Listen to instructions with browser TTS

## Tech Stack

- **Backend**: Python FastAPI
- **Vector Store**: FAISS (local)
- **Embeddings & LLM**: OpenAI API
- **Speech-to-Text**: OpenAI Whisper
- **Frontend**: React with Tailwind CSS

## Setup

### 1. Environment Variables

Set these in Replit Secrets or a `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

Optional settings:
```
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
WHISPER_API=true
```

### 2. Run the Application

The application runs both backend and frontend:

- Backend API runs on port 8000
- Frontend runs on port 5000 with proxy to backend

### 3. First Run - Data Ingestion

On first startup, the system automatically ingests sample scheme documents from `backend/sample_data/`. This creates:
- FAISS vector index
- Metadata JSON with scheme information

You can also trigger ingestion manually via `POST /api/ingest`.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and index status |
| `/transcribe` | POST | Transcribe audio file to text |
| `/query` | POST | Query schemes with text |
| `/ask-audio` | POST | Combined audio transcription + query |
| `/metadata` | GET | List all loaded schemes |
| `/ingest` | POST | Re-run document ingestion |

## Sample Schemes Included

1. **Crop Insurance Subsidy** - For small/marginal farmers
2. **Old Age Pension** - For elderly (60+) below poverty line
3. **Student Scholarship** - For SC/ST/OBC/Minority students
4. **Housing Subsidy** - PMAY housing assistance
5. **Micro Loan** - MUDRA loans for small businesses
6. **Disaster Relief** - For crop loss due to rain/flood

## Example Usage

**Voice Input**: "I am a farmer, my crop was destroyed by rain. I need help."

**Expected Response**:
- Disaster Relief for Crop Loss (high match)
- Crop Insurance Subsidy (medium match)
- Required documents listed
- Step-by-step next actions

## Privacy Notes

- Audio and transcripts are processed in memory only
- No permanent storage of user queries
- Logs are minimal for demo purposes

## Adding New Schemes

1. Add `.txt` or `.pdf` files to `backend/sample_data/`
2. Include clear sections for:
   - Eligibility criteria
   - Required documents
   - How to apply
3. Run `POST /ingest` to rebuild the index

## Token Costs

- Embeddings: ~$0.0001 per query
- LLM completion: ~$0.001-0.005 per query
- Whisper: ~$0.006 per minute of audio
