import os
import json
import shutil
import time
import sqlite3
import sys
from pathlib import Path

# Ensure all prints are flushed immediately
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode
from rems.llm.provider import LLMProvider

def run_continuous_simulation(num_chunks_to_process=3):
    """
    Continuous simulation runner that preserves state between runs.
    """
    base_dir = Path(__file__).parent / "outputs" / "continuous_run"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    state_file = base_dir / "simulation_state.json"
    db_path = base_dir / "rems_sim.db"
    chroma_path = base_dir / "chroma_sim"
    log_dir = base_dir / "llm_calls"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load state
    last_chunk_idx = 0
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                last_chunk_idx = state.get("last_chunk_idx", 0)
        except Exception as e:
            print(f"Warning: Could not load state file: {e}")

    # Initialize REMS
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path}",
            chromadb_path=str(chroma_path)
        ),
        embedding={"provider": "hash"},
        user_mode=UserMode.MULTI
    )
    
    # Setup LLM logging (similar to simulation.py)
    orig_complete = LLMProvider.complete
    call_counter = 0

    def complete_with_logging(self, task_type, messages, **kwargs):
        nonlocal call_counter
        call_counter += 1
        model = self._get_model(task_type)
        print(f"  [LLM] #{call_counter:03} {task_type:.<18} | {model:.<20}", end="", flush=True)
        
        t_start = time.perf_counter()
        content = orig_complete(self, task_type, messages, **kwargs)
        duration_ms = (time.perf_counter() - t_start) * 1000
        metrics = self._invocations[-1]
        
        # Log detail
        log_file = log_dir / f"chunk_{last_chunk_idx + 1}_{call_counter:03}_{task_type}.json"
        from dataclasses import asdict
        log_data = {
            "chunk_idx": last_chunk_idx + 1,
            "task_type": task_type,
            "model": model,
            "messages": messages,
            "response": content,
            "metrics": asdict(metrics)
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            
        print(f" DONE ({duration_ms:4.0f}ms)")
        return content

    LLMProvider.complete = complete_with_logging
    
    pipeline = REMSPipeline.from_config(config)
    
    # Load dataset
    dataset_path = Path(__file__).parents[3] / "data" / "hongloumeng_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    start_idx = last_chunk_idx
    end_idx = min(start_idx + num_chunks_to_process, len(chunks))

    print(f"\n{'='*60}")
    print(f" REMS CONTINUOUS SIMULATION: CHUNKS {start_idx + 1} TO {end_idx} ")
    print(f"{'='*60}")
    print(f"DB Path: {db_path}")
    
    # Get current shadow for display
    current_shadow = pipeline.meta_repo.get_shadow()
    print(f"Initial Shadow Length: {len(current_shadow.content)} chars")

    for i in range(start_idx, end_idx):
        chunk = chunks[i]
        content = chunk['content']
        print(f"\n--- PROCESSING CHUNK {i+1} (Length: {len(content)}) ---")
        print(f"  [DEBUG] Pipeline file: {pipeline.__class__.ingest.__code__.co_filename}")
        
        t_chunk_start = time.perf_counter()
        result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
        duration = time.perf_counter() - t_chunk_start
        
        print(f"  [OK] Duration: {duration:.2f}s")
        print(f"  [Events] Sealed: {len(result.sealed_events)}, Abstracted: {len(result.abstract_events)}")
        
        # Update and save state
        last_chunk_idx = i + 1
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({"last_chunk_idx": last_chunk_idx}, f, indent=2)

    total_events = pipeline.event_repo._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    final_shadow = pipeline.meta_repo.get_shadow()
    
    print(f"\n{'='*60}")
    print(f" RUN SUMMARY ")
    print(f"{'='*60}")
    print(f"Next Chunk to Process: {last_chunk_idx + 1}")
    print(f"Total Events in DB:    {total_events}")
    print(f"Final Shadow Length:   {len(final_shadow.content)} chars")
    print(f"Final Shadow Preview:  {final_shadow.content[:100]}...")

if __name__ == "__main__":
    n = 3
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except:
            pass
    run_continuous_simulation(n)
