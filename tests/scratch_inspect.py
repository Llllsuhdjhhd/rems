import sqlite3, json, sys, io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

db_path = "tests/hongloumeng_sim_5ch_1776595480/rems_sim.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# List tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = c.fetchall()
print("Tables:", [t[0] for t in tables])

for t in tables:
    tname = t[0]
    c.execute(f'SELECT COUNT(*) FROM "{tname}"')
    cnt = c.fetchone()[0]
    c.execute(f'PRAGMA table_info("{tname}")')
    cols = [r[1] for r in c.fetchall()]
    print(f"\n--- {tname} (count={cnt}) ---")
    print(f"  Columns: {cols}")

# Sample events
print("\n\n=== EVENTS SAMPLE (first 5) ===")
c.execute("SELECT event_id, summaries, role_list FROM events ORDER BY create_time LIMIT 5")
for eid, sums, roles in c.fetchall():
    s = json.loads(sums) if sums else {}
    r = json.loads(roles) if roles else []
    role_names = [x.get('name','?') for x in r]
    print(f"\nEvent: {eid}")
    print(f"  L1: {s.get('L1','N/A')[:120]}")
    print(f"  L2: {s.get('L2','N/A')[:120]}")
    print(f"  Roles: {role_names}")

# All events summary
print("\n\n=== ALL EVENTS OVERVIEW ===")
c.execute("SELECT event_id, summaries, is_abstract, abstraction_level FROM events ORDER BY create_time")
for eid, sums, is_abs, abs_lvl in c.fetchall():
    s = json.loads(sums) if sums else {}
    label = "[ABS]" if is_abs else "[EVT]"
    print(f"  {label} {eid} | L={abs_lvl} | {s.get('L1','N/A')[:80]}")

# Sample roles
print("\n\n=== ROLES (all) ===")
c.execute("SELECT role_id, name, entity_type, aliases FROM roles")
for rid, name, etype, aliases in c.fetchall():
    print(f"  {rid} | {name} | {etype} | aliases={aliases}")

# Semantic cards
print("\n\n=== SEMANTIC CARDS (first 5) ===")
c.execute("SELECT role_id, data FROM semantic_cards LIMIT 5")
for rid, data_json in c.fetchall():
    data = json.loads(data_json) if data_json else {}
    print(f"\nRole: {rid}")
    for k, v in data.items():
        if isinstance(v, str):
            print(f"  {k}: {v[:100]}")
        elif isinstance(v, dict):
            print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:150]}")

# White painting
print("\n\n=== WHITE PAINTING (first 10) ===")
c.execute("SELECT role_id, event_id, role_summary, importance FROM white_painting_entries ORDER BY create_time LIMIT 10")
for rid, eid, summary, imp in c.fetchall():
    print(f"  [{imp}] {rid} in {eid}: {summary[:100]}")

conn.close()
