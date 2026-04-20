
import json
import os
from pathlib import Path
from collections import Counter

def detailed_audit(path):
    log_dir = Path(path) / "llm_calls"
    task_counter = Counter()
    role_mentions = 0
    event_mentions = 0
    
    files = sorted(list(log_dir.glob("*.json")))
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                t_type = data.get("task_type", "unknown")
                task_counter[t_type] += 1
                
                # Check for role-specific identifiers in prompt or response
                messages = data.get("messages", [])
                full_text = "".join([m.get("content", "") for m in messages])
                
        except:
            continue
            
    return dict(task_counter)

ws_path = "tests/hongloumeng_sim_5ch_1776608561"
stats = detailed_audit(ws_path)
print(json.dumps(stats, indent=2))
