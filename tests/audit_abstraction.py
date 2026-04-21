import sqlite3
import json
import io
import sys

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

db_path = "tests/hongloumeng_sim_5ch_1776676690/rems_sim.db"
target_abstract_id = "EVT-45e964324ac444cda6cfdd6a2f8c7129"

def audit_abstraction():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Get the target abstraction details
    cursor.execute("SELECT source_events, summaries, insight FROM events WHERE event_id = ?", (target_abstract_id,))
    row = cursor.fetchone()
    if not row:
        print(f"Error: Could not find abstraction {target_abstract_id}")
        return
    
    source_ids = json.loads(row[0])
    target_summaries = json.loads(row[1])
    target_insight = row[2]
    
    print(f"--- TARGET ABSTRACTION: {target_abstract_id} ---")
    print(f"Theme: {target_summaries.get('L1', 'N/A')}")
    print(f"Insight: {target_insight}\n")
    
    # 2. List source events in detail
    print("--- INCLUDED SOURCE EVENTS ---")
    for sid in source_ids:
        cursor.execute("SELECT summaries FROM events WHERE event_id = ?", (sid,))
        srow = cursor.fetchone()
        if srow:
            ssum = json.loads(srow[0]).get('L1', 'N/A')
            print(f"[{sid}] {ssum}")
    
    # 3. List ALL other basic events and check for relevance
    print("\n--- EXCLUDED BASIC EVENTS AUDIT ---")
    cursor.execute("SELECT event_id, summaries FROM events WHERE is_abstract = 0")
    all_basics = cursor.fetchall()
    
    for eid, esum_json in all_basics:
        if eid in source_ids:
            continue
            
        esum = json.loads(esum_json).get('L1', 'N/A')
        # Check relevance: search for keywords related to Zhen, Jia, Fate, etc.
        relevant = False
        keywords = ["甄", "贾", "雨村", "士隐", "英莲", "娇杏", "封肃", "好了歌", "智通寺"]
        for kw in keywords:
            if kw in esum:
                relevant = True
                break
        
        status = "[RELEVANT?]" if relevant else "[UNRELATED]"
        print(f"{status} [{eid}] {esum}")

    conn.close()

if __name__ == "__main__":
    audit_abstraction()
