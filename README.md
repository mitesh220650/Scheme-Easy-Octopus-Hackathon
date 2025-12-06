Here is the professional, emoji-free version of the **README.md** file for your repository.

# Scheme-Easy (Local Octopus Hackathon Version)

A multilingual, voice-first chatbot designed to assist users in identifying government welfare schemes for which they qualify. This version is configured to execute entirely within a local environment using Ollama and local AI models, ensuring data privacy and zero operational costs.

## Features

- **Voice Input**: Processes spoken user input in their native language locally via the Whisper model.
- **Multilingual Support**: Automatically detects the input language and generates responses in the same language.
- **RAG Pipeline**: Utilizes FAISS for vector search and local embeddings to accurately match user situations with relevant schemes.
- **Privacy-First Architecture**: All data processing occurs on the local machine; no user audio or text data is transmitted to external servers.
- **Document Checklist**: Provides a precise list of required documents for each identified scheme.
- **Text-to-Speech**: Includes browser-based text-to-speech functionality for accessible information delivery.

## Technology Stack

- **Backend**: Python FastAPI
- **Vector Store**: FAISS (Local)
- **Embeddings**: `nomic-embed-text` (via Ollama)
- **LLM**: `llama3.2` (via Ollama)
- **Speech-to-Text**: OpenAI Whisper (Local `small` model)
- **Frontend**: React with Tailwind CSS

## Prerequisites

Ensure the following dependencies are installed before running the application:

1.  **Python 3.11+**
2.  **Node.js** (for frontend execution)
3.  **Ollama**: Required for running local LLMs. Download from [ollama.com](https://ollama.com).
4.  **FFmpeg**: Required for audio processing.
    * *macOS*: `brew install ffmpeg`
    * *Windows*: `choco install ffmpeg` (or download binaries manually).

---

## Setup Guide

### 1. Initialize Local AI Models

Ensure the Ollama service is running, then pull the required models using the following commands:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
````

### 2\. Configure Environment Variables

Create a file named `.env` in the `backend/` directory with the following configuration:

```env
# Placeholder key for library compatibility
OPENAI_API_KEY=ollama

# Connection string for local Ollama instance
OPENAI_BASE_URL=http://localhost:11434/v1

# Local Model Configurations
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2

# System Configuration
VECTOR_DB=faiss
WHISPER_API=false  # Forces usage of the local Whisper model
```

### 3\. Backend Initialization

Open a terminal, navigate to the `backend/` directory, and execute the following:

```bash
# Install Python dependencies
pip install fastapi uvicorn openai numpy python-multipart pdfplumber faiss-cpu python-dotenv openai-whisper

# 1. Execute Data Ingestion (Constructs the vector search index)
# Run this once initially or whenever new scheme documents are added.
python ingest.py

# 2. Launch the Application Server
# Note: On macOS, if an OpenMP error occurs, export the following variable before running:
export KMP_DUPLICATE_LIB_OK=TRUE
python app.py
```

*The backend server will initialize at `http://127.0.0.1:8000`.*

### 4\. Frontend Initialization

Open a new terminal, navigate to the `frontend-react/` directory, and execute the following:

```bash
# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

*Access the application via the browser at the URL provided (typically `http://localhost:5173`).*

-----

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns health status and index state. |
| `/transcribe` | POST | Transcribes uploaded audio files using the local Whisper model. |
| `/query` | POST | Queries the RAG pipeline using the Ollama LLM. |
| `/ask-audio` | POST | Handles combined audio transcription and query processing. |
| `/metadata` | GET | Retrieves a list of all ingested schemes. |
| `/ingest` | POST | Triggers the document ingestion and indexing process manually. |

## Adding New Schemes

1.  Place `.txt` or `.pdf` files into the `backend/sample_data/` directory.
2.  Ensure the document content explicitly details **Eligibility Criteria**, **Required Documents**, and **Benefits**.
3.  Execute `python ingest.py` in the backend directory to rebuild the FAISS index.

## Troubleshooting

  - **OpenMP Error (macOS):** If the application crashes with an OMP error, execute `export KMP_DUPLICATE_LIB_OK=TRUE` in the terminal before running `app.py`.
  - **Slow Transcription:** The Whisper model (\~500MB) is downloaded during the first execution of the voice input feature. Subsequent executions will be significantly faster.
  - **"Model not found" Error:** Verify that the `ollama pull` commands listed in the Setup Guide were executed successfully and that Ollama is running.
  - **Microphone Access:** Ensure the web browser has been granted permission to access the microphone.

## Resource Usage

  - **Embeddings:** Zero cost (Local Execution)
  - **LLM:** Zero cost (Local Execution)
  - **Voice-to-Text:** Zero cost (Local Execution)

<!-- end list -->

```
```
