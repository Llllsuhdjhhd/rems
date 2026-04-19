import json
import uuid
from rems.config import REMSConfig
from rems.pipeline import REMSPipeline

def main():
    # Configure REMS to use a temporary SQLite DB
    config = REMSConfig()
    config.storage.database_url = "sqlite:///rems_sim_hlm_test.db"
    
    # Initialize pipeline
    print("Initializing Pipeline...")
    pipeline = REMSPipeline.from_config(config)
    
    # Load dataset
    print("Loading data...")
    with open("data/hongloumeng_dataset.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    if not chunks:
        print("No chunks found")
        return
        
    chunk_0 = chunks[0]["content"]
    print(f"----- Read Chunk 0 (length: {len(chunk_0)}) -----")
    
    # Process through pipeline
    print("Ingesting chunk...")
    input_id = f"TEST-INPUT-{uuid.uuid4().hex[:8]}"
    result = pipeline.ingest(chunk_0, force_save=True, input_id=input_id)
    
    # Print results
    print(f"\n----- INGESTION RESULTS -----")
    print(f"Input ID sent: {input_id}")
    print(f"Sealed Events: {len(result.sealed_events)}")
    for ev in result.sealed_events:
        print(f"\nEVENT ID: {ev.event_id}")
        print(f"input_id: {ev.input_id}")
        print(f"create_time: {ev.create_time}")
        print(f"Affective Energy (AE): {ev.affective_energy:.2f}")
        print(f"L1 Summary:\n{ev.summaries.get('L1', '')}")
        print("\nROLES IN EVENT:")
        for r_entry in ev.role_list:
            # Query the actual role to see is_suspicious
            role_obj = pipeline.role_repo.get(r_entry.role_id)
            suspicious_flag = getattr(role_obj, 'is_suspicious', False) if role_obj else False
            
            # Print role information
            print(f"  - {r_entry.role_id} (Importance: {r_entry.importance.value if hasattr(r_entry.importance, 'value') else r_entry.importance})")
            print(f"    Name: {role_obj.name if role_obj else 'Unknown'} | Suspicious: {suspicious_flag}")
            
            # Print Vedana / Klesha to check default 1.0 base
            v = r_entry.emotional_model.vedana
            print(f"    Vedana: joy={v.joy}, suffering={v.suffering}")

if __name__ == "__main__":
    main()
