import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from openai import OpenAI

MODELS_DIR = Path(__file__).parent / "models"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

SYSTEM_PROMPT = """You are Scheme-Easy assistant. Task: given a user's situation and a set of retrieved document excerpts describing government welfare schemes, produce:

1) A short direct answer (1-3 lines) stating exactly which schemes the user appears to qualify for. Use confident but cautious language (e.g., "You *may* qualify for...") when uncertain.

2) For each suggested scheme, list the **exact documents** required in simple words (no legal jargon), and where the user can obtain them if applicable, and any likely next steps (e.g., "Visit Gram Panchayat office", "Apply online at example.gov.in").

3) A one-line "why this fits" explanation (which criteria from the retrieved text matched the user's situation).

4) All text should be simple, suitable for low-literacy users. Keep sentences short. Use the user's language; if not specified use English. If you detect a different language in the input, reply in that language.

5) Output in strict JSON format with fields: "summary", "schemes": [ { "title", "why", "documents":[...], "next_steps": [...], "source_file", "confidence" } ], and "follow_up_questions" (list of at most 3 clarifying questions). Do not include any other fields.

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no explanation text outside the JSON."""


class RAGPipeline:
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.index = None
        self.embeddings: Optional[List[List[float]]] = None
        self.metadata: Optional[List[Dict]] = None
        self._loaded = False
    
    def _get_client(self) -> OpenAI:
        if self.client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = OpenAI(api_key=api_key)
        return self.client
    
    def load(self) -> bool:
        if self._loaded:
            return True
        
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            return False
        
        try:
            with open(FAISS_INDEX_PATH, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data.get('embeddings', [])
                if 'index' in data and faiss is not None:
                    self.index = faiss.deserialize_index(data['index'])
            
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)
            
            self._loaded = True
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def get_query_embedding(self, text: str) -> List[float]:
        client = self._get_client()
        model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        
        response = client.embeddings.create(
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self._loaded:
            if not self.load():
                return []
        
        query_embedding = self.get_query_embedding(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        
        if self.index is not None and faiss is not None:
            faiss.normalize_L2(query_array)
            scores, indices = self.index.search(query_array, min(top_k, len(self.metadata)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.metadata):
                    results.append((self.metadata[idx], float(score)))
            return results
        else:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            
            query_norm = query_array / np.linalg.norm(query_array)
            embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            
            similarities = np.dot(embeddings_norm, query_norm.T).flatten()
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.metadata):
                    results.append((self.metadata[idx], float(similarities[idx])))
            return results
    
    def build_user_prompt(self, user_text: str, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        prompt = f'User situation:\n"{user_text}"\n\n'
        prompt += "Retrieved documents (each labeled with source filename and excerpt). Use ONLY the information in these excerpts to decide relevant schemes:\n\n"
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            prompt += f"[ DOC {i}: {doc['file_name']} (relevance: {score:.2f}) ]\n"
            prompt += f'"{doc["chunk_text"]}"\n\n'
        
        prompt += "Now produce the JSON described above. If information is missing to decide eligibility, ask 1-3 follow-up questions in the follow_up_questions array."
        
        return prompt
    
    def generate_response(self, user_text: str, retrieved_docs: List[Tuple[Dict, float]], language: str = "auto") -> Dict[str, Any]:
        client = self._get_client()
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        
        user_prompt = self.build_user_prompt(user_text, retrieved_docs)
        
        if language != "auto":
            system_prompt = SYSTEM_PROMPT + f"\n\nIMPORTANT: Respond in {language} language."
        else:
            system_prompt = SYSTEM_PROMPT
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            
            result["llm_tokens"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            return result
            
        except json.JSONDecodeError:
            return {
                "summary": "I found some relevant schemes but had trouble formatting the response. Please try again.",
                "schemes": [],
                "follow_up_questions": ["Could you describe your situation in more detail?"],
                "error": "JSON parsing failed",
                "raw_response": content if 'content' in dir() else None
            }
        except Exception as e:
            return {
                "summary": f"An error occurred: {str(e)}",
                "schemes": [],
                "follow_up_questions": [],
                "error": str(e)
            }
    
    def query(self, text: str, top_k: int = 5, language: str = "auto") -> Dict[str, Any]:
        retrieved_docs = self.retrieve(text, top_k)
        
        if not retrieved_docs:
            return {
                "summary": "No matching schemes found. Please provide more details about your situation.",
                "schemes": [],
                "follow_up_questions": [
                    "What is your occupation or main source of income?",
                    "What kind of help are you looking for?",
                    "Which state or district do you live in?"
                ],
                "raw_retrieved": []
            }
        
        response = self.generate_response(text, retrieved_docs, language)
        
        response["raw_retrieved"] = [
            {
                "file_name": doc["file_name"],
                "scheme_title": doc.get("scheme_title", ""),
                "chunk_text": doc["chunk_text"][:200] + "...",
                "similarity": score
            }
            for doc, score in retrieved_docs
        ]
        
        return response
    
    def get_metadata_summary(self) -> List[Dict[str, Any]]:
        if not self._loaded:
            if not self.load():
                return []
        
        schemes = {}
        for doc in self.metadata:
            doc_id = doc["doc_id"]
            if doc_id not in schemes:
                schemes[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": doc["file_name"],
                    "scheme_title": doc.get("scheme_title", ""),
                    "eligibility_bullets": doc.get("eligibility_bullets", []),
                    "total_chunks": doc.get("total_chunks", 1)
                }
        
        return list(schemes.values())


rag_pipeline = RAGPipeline()


def query(text: str, top_k: int = 5, language: str = "auto") -> Dict[str, Any]:
    return rag_pipeline.query(text, top_k, language)


def get_metadata() -> List[Dict[str, Any]]:
    return rag_pipeline.get_metadata_summary()


def is_loaded() -> bool:
    return rag_pipeline._loaded or rag_pipeline.load()
