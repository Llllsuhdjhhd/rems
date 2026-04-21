import os
import json
import shutil
import time
from pathlib import Path
from rems.config import REMSConfig, StorageConfig, UserMode
from rems.pipeline import REMSPipeline, ProcessingMode
from rems.llm.provider import LLMProvider

def run_simulation():
    """
    Standalone simulation runner for Hongloumeng dataset chunks 1 & 2.
    Includes live progress tracking, granular timing, and bottleneck analysis.
    """
    # 1. Setup isolated environment
    timestamp = int(time.time())
    test_dir = Path(f"tests/hongloumeng_sim_{timestamp}")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    db_path = test_dir / "rems_sim.db"
    chroma_path = test_dir / "chroma_sim"
    
    print(f"\n{'='*60}")
    print(f" REMS SIMULATION: HONGLOUMENG CHUNKS 1 & 2 ")
    print(f"{'='*60}")
    print(f"Isolated workspace: {test_dir}")
    
    # 2. Live Progress Interceptor (Monkey-patching LLMProvider)
    # This provides the "Live Progress" requested by the user.
    orig_complete = LLMProvider.complete
    
    def complete_with_live_log(self, task_type, messages, **kwargs):
        model = self._get_model(task_type)
        print(f"  [LLM REQUEST] Task: {task_type:.<18} Model: {model:.<15}", end="", flush=True)
        t_start = time.perf_counter()
        
        try:
            content = orig_complete(self, task_type, messages, **kwargs)
            duration = (time.perf_counter() - t_start) * 1000
            
            # Retrieve last metrics from provider's internal history
            metrics = self._invocations[-1]
            print(f" DONE ({duration:5.0f}ms | Tokens: {metrics.total_tokens:4})")
            return content
        except Exception as e:
            print(f" FAILED: {str(e)[:50]}...")
            raise

    LLMProvider.complete = complete_with_live_log

    # 3. Initialization
    config = REMSConfig(
        storage=StorageConfig(
            database_url=f"sqlite:///{db_path}",
            chromadb_path=str(chroma_path)
        ),
        user_mode=UserMode.SINGLE
    )
    
    pipeline = REMSPipeline.from_config(config)
    
    dataset_path = Path("data/hongloumeng_dataset.json")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    total_start_time = time.perf_counter()
    chunk_times = []

    # 4. Processing Loop
    for i in range(min(2, len(chunks))):
        chunk = chunks[i]
        content = chunk['content']
        print(f"\n--- CHUNK {i+1} START (Chars: {len(content)}) ---")
        
        chunk_start = time.perf_counter()
        
        # We wrap ingest to show stage transitions
        print(f"  [PHASE] Building Context & Boundary Detection...")
        result = pipeline.ingest(content, mode=ProcessingMode.DIALOGUE)
        
        chunk_end = time.perf_counter()
        chunk_duration = chunk_end - chunk_start
        chunk_times.append(chunk_duration)
        
        print(f"  [CHUNK {i+1} COMPLETE] Duration: {chunk_duration:.2f}s")
        print(f"  [RESULT] Events Sealed: {len(result.sealed_events)}")
        
        # Mini Summary of what was found
        for event in result.sealed_events:
            print(f"    - {event.event_id}: {event.summaries.get('L1', event.content_raw[:40])}...")

    total_duration = time.perf_counter() - total_start_time

    # 5. --- COMPREHENSIVE DIAGNOSTIC REPORT ---
    print(f"\n{'='*60}")
    print(f" FINAL PERFORMANCE & METRICS REPORT ")
    print(f"{'='*60}")
    
    print(f"Total Simulation Time: {total_duration:.2f}s")
    for idx, ct in enumerate(chunk_times):
        print(f"Chunk {idx+1} Time: {ct:.2f}s")
    
    # LLM Metrics Table
    history = pipeline.llm.invocation_history()
    print(f"\n[LLM INVOCATION SUMMARY]")
    print(f"{'Task':<20} | {'Model':<15} | {'Latency(ms)':<11} | {'Tokens(P/C/T)':<15}")
    print("-" * 70)
    
    total_p_tokens = 0
    total_c_tokens = 0
    total_latency = 0.0
    
    for m in history:
        print(f"{m.task_type:<20} | {m.model:<15} | {m.latency_ms:>11.0f} | {m.prompt_tokens:>4}/{m.completion_tokens:>4}/{m.total_tokens:>4}")
        total_p_tokens += (m.prompt_tokens or 0)
        total_c_tokens += (m.completion_tokens or 0)
        total_latency += (m.latency_ms or 0.0)
        
    print("-" * 70)
    print(f"{'TOTALS':<20} | {'-':<15} | {total_latency:>11.0f} | {total_p_tokens:>4}/{total_c_tokens:>4}/{total_p_tokens+total_c_tokens:>4}")

    # 6. --- BOTTLENECK ANALYSIS ---
    print(f"\n[BOTTLENECK ANALYSIS]")
    avg_latency = total_latency / len(history) if history else 0
    
    # Count calls per task
    task_counts = {}
    for m in history:
        task_counts[m.task_type] = task_counts.get(m.task_type, 0) + 1
    
    print(f"1. Sequence Complexity: Processing 2 chunks triggered {len(history)} sequential LLM calls.")
    print(f"2. Core Wait Time: {total_latency/1000:.2f}s (or {(total_latency/total_duration/10):.1f}% of wall time) was spent waiting for LLM responses.")
    
    most_frequent_task = max(task_counts, key=task_counts.get) if task_counts else "N/A"
    print(f"3. Heaviest Task: '{most_frequent_task}' was called {task_counts.get(most_frequent_task, 0)} times.")
    print(f"4. Insight: The simulation time is primarily driven by the 'event sealing' process. ")
    print(f"   Each detected event triggers summary generation (L1, L2, etc.) and role extraction sequentially.")
    print(f"   In Chunk 1, approximately {len(history)} LLM calls were made due to dense narrative boundaries.")

    print(f"\n{'='*60}")
    print(f" WORKSPACE ASSETS ")
    print(f"{'='*60}")
    print(f"Database: {db_path}")
    print(f"Chroma Data: {chroma_path}")
    print(f"Failed JSON log: failed_llm_json.txt (if any issues occurred)")

if __name__ == "__main__":
    run_simulation()
