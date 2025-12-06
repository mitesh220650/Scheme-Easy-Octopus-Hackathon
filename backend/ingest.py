import os
import json
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from openai import OpenAI

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"
MODELS_DIR = Path(__file__).parent / "models"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def extract_text_from_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
            except ImportError:
                print(f"Warning: Cannot read PDF {file_path}. Install pdfplumber or PyPDF2.")
                return ""
    return ""


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def extract_eligibility_bullets(text: str) -> List[str]:
    bullets = []
    patterns = [
        r'[-•*]\s*(.+?)(?=[-•*]|\n\n|$)',
        r'\d+\.\s*(.+?)(?=\d+\.|$)',
    ]
    
    eligibility_keywords = [
        'eligib', 'who can apply', 'required documents', 'documents required',
        'must be', 'should be', 'criteria', 'qualification', 'proof'
    ]
    
    lines = text.split('\n')
    in_eligibility_section = False
    
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in eligibility_keywords):
            in_eligibility_section = True
        
        if in_eligibility_section:
            line = line.strip()
            if line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', line):
                clean_line = re.sub(r'^[-•*\d.]+\s*', '', line).strip()
                if clean_line and len(clean_line) > 10:
                    bullets.append(clean_line)
            if line == '' or line.startswith('#') or ':' in line and len(line) < 50:
                if len(bullets) > 3:
                    in_eligibility_section = False
    
    return bullets[:10]


def extract_scheme_title(text: str, filename: str) -> str:
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) > 5 and len(line) < 100:
            if line.isupper() or line[0].isupper():
                return line
    
    title = filename.replace('_', ' ').replace('.txt', '').replace('.pdf', '')
    return title.title()


def get_embeddings(texts: List[str], client: OpenAI) -> List[List[float]]:
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    
    embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        for item in response.data:
            embeddings.append(item.embedding)
    
    return embeddings


def build_faiss_index(embeddings: List[List[float]]) -> Any:
    if faiss is None:
        return None
    
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    return index


def run_ingestion() -> Dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    client = get_openai_client()
    
    all_chunks = []
    metadata = []
    
    sample_files = list(SAMPLE_DATA_DIR.glob("*.txt")) + list(SAMPLE_DATA_DIR.glob("*.pdf"))
    
    if not sample_files:
        return {"status": "error", "message": "No sample data files found"}
    
    print(f"Found {len(sample_files)} files to process")
    
    for file_path in sample_files:
        print(f"Processing: {file_path.name}")
        
        text = extract_text_from_file(file_path)
        if not text:
            continue
        
        text = normalize_text(text)
        chunks = chunk_text(text)
        
        scheme_title = extract_scheme_title(text, file_path.name)
        eligibility_bullets = extract_eligibility_bullets(text)
        
        for chunk_id, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "doc_id": file_path.stem,
                "file_name": file_path.name,
                "chunk_id": chunk_id,
                "chunk_text": chunk,
                "scheme_title": scheme_title,
                "eligibility_bullets": eligibility_bullets,
                "total_chunks": len(chunks)
            })
    
    if not all_chunks:
        return {"status": "error", "message": "No text extracted from files"}
    
    print(f"Generated {len(all_chunks)} chunks, creating embeddings...")
    
    embeddings = get_embeddings(all_chunks, client)
    
    print("Building FAISS index...")
    
    index = build_faiss_index(embeddings)
    
    if index is not None:
        with open(FAISS_INDEX_PATH, 'wb') as f:
            pickle.dump({
                'index': faiss.serialize_index(index),
                'embeddings': embeddings
            }, f)
    else:
        with open(FAISS_INDEX_PATH, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings
            }, f)
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Ingestion complete! Processed {len(sample_files)} files, {len(all_chunks)} chunks")
    
    return {
        "status": "success",
        "files_processed": len(sample_files),
        "chunks_created": len(all_chunks),
        "index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(METADATA_PATH)
    }


def is_index_available() -> bool:
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()


if __name__ == "__main__":
    result = run_ingestion()
    print(json.dumps(result, indent=2))
