import sys
import time

def trace_import(module_path):
    t0 = time.perf_counter()
    print(f"DEBUG: Importing {module_path}...", end="", flush=True)
    __import__(module_path)
    print(f" SUCCESS ({time.perf_counter()-t0:.4f}s)")

modules = [
    "logging",
    "dataclasses",
    "rems.config",
    "rems.llm.provider",
    "rems.models.event",
    "rems.storage.database",
    "rems.storage.vector_store",
    "rems.skills.boundary_detection",
    "rems.skills.role_extraction",
    "rems.services.event_service",
    "rems.services.metabolism_service",
    "rems.pipeline"
]

print("Starting import trace...")
for m in modules:
    try:
        trace_import(m)
    except Exception as e:
        print(f" ERROR: {e}")
print("Trace complete.")
