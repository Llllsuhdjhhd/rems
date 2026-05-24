import sys
import os
import json
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add src to path
sys.path.append(os.path.abspath("src"))

# Mock embedding function BEFORE importing VectorStore
import chromadb.utils.embedding_functions as ef
class DummyEF:
    def __call__(self, input: list[str]):
        return [[0.0] * 384 for _ in input]
    def embed_query(self, input: str):
        return [0.0] * 384
    def embed_documents(self, input: list[str]):
        return [[0.0] * 384 for _ in input]
    def name(self):
        return "DummyEF"
ef.SentenceTransformerEmbeddingFunction = MagicMock(return_value=DummyEF())

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline
from rems.llm.provider import LLMProvider

# Setup metrics logging
LOG_DIR = Path("tests/benchmark_after_refactor")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_benchmark(max_chunks=3):
    print(f"============================================================")
    print(f" REMS THROUGHPUT BENCHMARK (3 ROUNDS) ")
    print(f"============================================================")
    
    # 1. Custom logging for LLM calls
    orig_complete = LLMProvider.complete
    call_counter = 0

    def complete_with_logging(self, task_type, messages, **kwargs):
        nonlocal call_counter
        call_counter += 1
        current_call_id = call_counter
        
        model = self._get_model(task_type)
        print(f"  [LLM] #{current_call_id:03} {task_type:.<18} | {model:.<20}", end="", flush=True)
        
        t_start = time.perf_counter()
        try:
            content = orig_complete(self, task_type, messages, **kwargs)
            duration_ms = (time.perf_counter() - t_start) * 1000
            metrics = self._invocations[-1]
            
            log_file = LOG_DIR / f"{current_call_id:03}_{task_type}_{model}.json"
            log_data = {
                "call_id": current_call_id,
                "task_type": task_type,
                "model": model,
                "messages": messages,
                "response": content,
                "metrics": {
                    "latency_ms": duration_ms,
                    "prompt_tokens": metrics.prompt_tokens,
                    "completion_tokens": metrics.completion_tokens,
                    "total_tokens": metrics.total_tokens
                }
            }
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f" DONE ({duration_ms:5.0f}ms | Completion Tokens: {metrics.completion_tokens:4})")
            return content
        except Exception as e:
            print(f" FAILED: {str(e)[:50]}")
            raise

    LLMProvider.complete = complete_with_logging

    # 2. Initialization
    db_path = LOG_DIR / "benchmark.db"
    if db_path.exists():
        db_path.unlink()
        
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path}",
            chromadb_path=str(LOG_DIR / "chroma_sim")
        ),
        user_mode=UserMode.MULTI
    )
    # Ensure flash models for speed and reliability in this test
    config.llm.task_models.default = "qwen3.5-flash"
    config.llm.task_models.summary = "qwen3.5-flash"
    config.llm.task_models.boundary_detection = "qwen3.5-flash"
    config.llm.task_models.role_extraction = "qwen3.5-flash"

    pipeline = REMSPipeline.from_config(config)
    
    dataset_path = Path("data/hongloumeng_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # 3. Processing Loop
    for i in range(min(max_chunks, len(chunks))):
        chunk = chunks[i]
        content = chunk['content']
        print(f"\n--- CHUNK {i+1} START (Chars: {len(content)}) ---")
        pipeline.ingest(content)
        print(f"--- CHUNK {i+1} END ---")

if __name__ == "__main__":
    run_benchmark(3)
