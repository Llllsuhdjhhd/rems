
import json
import os
from pathlib import Path

def audit_workspace(path):
    log_dir = Path(path) / "llm_calls"
    total_prompt = 0
    total_completion = 0
    total_latency = 0
    count = 0
    
    if not log_dir.exists():
        return None
        
    for f in log_dir.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                metrics = data.get("metrics", {})
                total_prompt += metrics.get("prompt_tokens", 0)
                total_completion += metrics.get("completion_tokens", 0)
                total_latency += metrics.get("latency_ms", 0)
                count += 1
        except:
            continue
            
    return {
        "count": count,
        "prompt": total_prompt,
        "completion": total_completion,
        "latency_sec": total_latency / 1000.0,
        "avg_completion": total_completion / count if count > 0 else 0
    }

ws_old = "tests/hongloumeng_sim_5ch_1776605544" # 全文本复读版
ws_new = "tests/hongloumeng_sim_5ch_1776607593" # 指针式物理切片版

stats_old = audit_workspace(ws_old)
stats_new = audit_workspace(ws_new)

print(json.dumps({"old": stats_old, "new": stats_new}, indent=2))
