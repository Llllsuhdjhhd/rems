import chromadb
from pathlib import Path

# Path to the simulation chroma data
chroma_path = Path("tests/scenarios/hongloumeng/outputs/sim_runs/sim_run_1777361998/chroma_sim")
client = chromadb.PersistentClient(path=str(chroma_path))
collection = client.get_collection("rems_events")

# Get one item with metadata
results = collection.get(limit=1, include=["metadatas"])
print(results["metadatas"])
