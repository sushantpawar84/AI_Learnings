"""
Query and Inference Interface
Unified interface for querying the model with different question types.
Includes RAG (Retrieval-Augmented Generation) and source attribution.
"""

import json
import torch
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np


class QueryType(Enum):
    """Types of queries the system supports"""
    EXPLAIN_METHOD = "explain_method"
    EXPLAIN_CLASS = "explain_class"
    EXPLAIN_FILE = "explain_file"
    EXPLAIN_LOGIC = "explain_logic"
    FIND_RELATED = "find_related"
    CUSTOM = "custom"


@dataclass
class QueryResponse:
    """Response from the model"""
    query: str
    query_type: QueryType
    response: str
    confidence: float  # 0-1, higher is more confident
    source_citations: List[Dict[str, Any]]  # Which code chunks were referenced
    generation_time_ms: float
    model_version: int
    timestamp: str


class CodeRetriever:
    """Retrieves relevant code snippets using semantic search"""

    def __init__(self, chunks_file: str):
        self.chunks = self._load_chunks(chunks_file)
        self.embeddings_cache = {}

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query"""

        # Simple keyword matching for MVP (can be replaced with semantic search)
        relevant = []
        query_terms = set(query.lower().split())

        for chunk in self.chunks:
            # Calculate relevance score
            score = 0
            chunk_text = f"{chunk['chunk_name']} {chunk['description']}".lower()
            chunk_terms = set(chunk_text.split())

            # Jaccard similarity
            if chunk_terms:
                intersection = query_terms & chunk_terms
                union = query_terms | chunk_terms
                score = len(intersection) / len(union) if union else 0

            if score > 0:
                relevant.append({
                    "chunk": chunk,
                    "score": score
                })

        # Sort by score and return top-k
        relevant.sort(key=lambda x: x['score'], reverse=True)
        return [item['chunk'] for item in relevant[:top_k]]

    def _load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from file"""
        chunks = []
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk = json.loads(line)
                        chunks.append(chunk)
                    except:
                        continue
        except Exception as e:
            print(f"Error loading chunks: {e}")
        return chunks


class QueryBuilder:
    """Builds prompts for different query types"""

    @staticmethod
    def build_explain_method_prompt(method_name: str, class_name: str = None) -> str:
        """Build prompt for explaining a method"""
        if class_name:
            return f"Explain what the method '{method_name}' does in the class '{class_name}'."
        else:
            return f"Explain what the method '{method_name}' does."

    @staticmethod
    def build_explain_class_prompt(class_name: str) -> str:
        """Build prompt for explaining a class"""
        return f"What is the purpose of the '{class_name}' class and what are its main responsibilities?"

    @staticmethod
    def build_explain_file_prompt(file_path: str) -> str:
        """Build prompt for explaining a file"""
        file_name = Path(file_path).name
        return f"Summarize the purpose and main components of the file '{file_name}'."

    @staticmethod
    def build_explain_logic_prompt(code_snippet: str) -> str:
        """Build prompt for explaining code logic"""
        return f"Explain the logic of this code snippet:\n\n{code_snippet}"

    @staticmethod
    def build_find_related_prompt(entity_name: str) -> str:
        """Build prompt for finding related code"""
        return f"What other methods, classes, or files are related to '{entity_name}'?"


class CodeModelInference:
    """Inference engine for code understanding model"""

    def __init__(self, model_path: str, tokenizer_path: str,
                 chunks_file: str = None, use_lora: bool = True):
        print(f"Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Load LoRA weights if applicable
        if use_lora:
            try:
                self.model = PeftModel.from_pretrained(self.model, model_path)
                print("LoRA weights loaded")
            except:
                print("LoRA loading failed, using base model")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        # Initialize retriever if chunks file provided
        self.retriever = CodeRetriever(chunks_file) if chunks_file else None
        self.query_builder = QueryBuilder()

    def query(self, question: str, query_type: QueryType = QueryType.CUSTOM,
             use_rag: bool = True, max_length: int = 200) -> QueryResponse:
        """Process a query and return response"""

        import time
        start_time = time.time()

        # Retrieve context if RAG enabled
        context = ""
        citations = []
        if use_rag and self.retriever:
            relevant_chunks = self.retriever.retrieve_relevant_chunks(question, top_k=2)
            citations = [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "file": chunk.get("file_path"),
                    "chunk_type": chunk.get("chunk_type"),
                    "name": chunk.get("chunk_name"),
                    "relevance": "high"
                }
                for chunk in relevant_chunks
            ]

            if relevant_chunks:
                context = "RELEVANT CODE CONTEXT:\n"
                for chunk in relevant_chunks:
                    context += f"\n{chunk.get('chunk_type').upper()}: {chunk.get('chunk_name')}\n"
                    context += f"{chunk.get('code_content', '')[:300]}...\n"

        # Build full prompt
        full_prompt = f"{context}\n\nQUESTION: {question}\n\nANSWER:"

        # Generate response
        with torch.no_grad():
            inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer from response
        if "ANSWER:" in response_text:
            answer = response_text.split("ANSWER:")[-1].strip()
        else:
            answer = response_text.strip()

        # Calculate confidence (simple heuristic)
        confidence = min(1.0, len(citations) * 0.3 + 0.4)

        generation_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            query=question,
            query_type=query_type,
            response=answer,
            confidence=confidence,
            source_citations=citations,
            generation_time_ms=generation_time_ms,
            model_version=1,
            timestamp=datetime.now().isoformat()
        )

    def explain_method(self, method_name: str, class_name: str = None) -> QueryResponse:
        """Explain what a method does"""
        prompt = self.query_builder.build_explain_method_prompt(method_name, class_name)
        return self.query(prompt, QueryType.EXPLAIN_METHOD)

    def explain_class(self, class_name: str) -> QueryResponse:
        """Explain what a class does"""
        prompt = self.query_builder.build_explain_class_prompt(class_name)
        return self.query(prompt, QueryType.EXPLAIN_CLASS)

    def explain_file(self, file_path: str) -> QueryResponse:
        """Explain what a file does"""
        prompt = self.query_builder.build_explain_file_prompt(file_path)
        return self.query(prompt, QueryType.EXPLAIN_FILE)

    def explain_code(self, code_snippet: str) -> QueryResponse:
        """Explain a code snippet"""
        prompt = self.query_builder.build_explain_logic_prompt(code_snippet)
        return self.query(prompt, QueryType.EXPLAIN_LOGIC)

    def find_related_code(self, entity_name: str) -> QueryResponse:
        """Find code related to an entity"""
        prompt = self.query_builder.build_find_related_prompt(entity_name)
        return self.query(prompt, QueryType.FIND_RELATED)


class QueryCache:
    """Simple caching layer for frequent queries"""

    def __init__(self, cache_file: str = None):
        self.cache = {}
        self.cache_file = cache_file
        if cache_file and Path(cache_file).exists():
            self._load_cache()

    def get(self, query: str) -> Optional[QueryResponse]:
        """Retrieve cached response"""
        query_hash = hash(query.lower())
        return self.cache.get(query_hash)

    def set(self, query: str, response: QueryResponse):
        """Cache response"""
        query_hash = hash(query.lower())
        self.cache[query_hash] = response
        if self.cache_file:
            self._save_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                cached = json.load(f)
                for key, value_dict in cached.items():
                    self.cache[int(key)] = QueryResponse(**value_dict)
        except Exception as e:
            print(f"Error loading cache: {e}")

    def _save_cache(self):
        try:
            cache_dict = {str(k): asdict(v) for k, v in self.cache.items()}
            with open(self.cache_file, 'w') as f:
                json.dump(cache_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")


class CodeExplainerAPI:
    """High-level API for code explanation"""

    def __init__(self, model_path: str, tokenizer_path: str, chunks_file: str = None):
        self.inference = CodeModelInference(model_path, tokenizer_path, chunks_file)
        self.cache = QueryCache()

    def explain(self, query: str, use_cache: bool = True) -> QueryResponse:
        """Explain code or answer questions about it"""

        # Check cache first
        if use_cache:
            cached = self.cache.get(query)
            if cached:
                print("(Returned from cache)")
                return cached

        # Generate response
        response = self.inference.query(query, use_rag=True)

        # Cache response
        if use_cache:
            self.cache.set(query, response)

        return response

    def print_response(self, response: QueryResponse):
        """Pretty print a response"""
        print("\n" + "="*60)
        print(f"Query: {response.query}")
        print(f"Type: {response.query_type.value}")
        print(f"Confidence: {response.confidence:.2%}")
        print("="*60)
        print(f"\nResponse:\n{response.response}")

        if response.source_citations:
            print(f"\nCitations:")
            for citation in response.source_citations:
                print(f"  - {citation['chunk_type'].upper()}: {citation['name']} ({citation['file']})")

        print(f"\nGeneration time: {response.generation_time_ms:.2f}ms")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    # api = CodeExplainerAPI(
    #     model_path="./models/code-gpt2/checkpoint-v1",
    #     tokenizer_path="./models/code-gpt2/checkpoint-v1",
    #     chunks_file="./datasets/chunks.jsonl"
    # )
    #
    # response = api.explain("What does the UserController class do?")
    # api.print_response(response)
    pass



