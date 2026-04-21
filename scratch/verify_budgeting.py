import sys
import os
from math import exp

# Add src to path
sys.path.append(os.path.abspath("src"))

from rems.utils.text import segment_sentences, format_indexed_text, decode_indices
from rems.config import REMSConfig
from rems.services.event_service import EventService
from rems.storage.repository import EventRepository
from rems.storage.database import Database

def test_utils():
    print("Testing Utils...")
    text = "甄士隐递英莲于贾雨村。贾雨村笑着接了。！真的吗？是的；就这样。"
    sentences = segment_sentences(text)
    print(f"Sentences: {sentences}")
    indexed = format_indexed_text(sentences)
    print("Indexed prompt text:")
    print(indexed)
    
    indices = [1, 3, 5]
    decoded = decode_indices(sentences, indices)
    print(f"Decoded [1, 3, 5]: {decoded}")
    expected = "甄士隐递英莲于贾雨村。！就这样。" # Wait, segmenting with regex might be tricky
    print(f"Expected approx: {expected}")

def test_budgeting():
    print("\nTesting Budgeting...")
    cfg = REMSConfig()
    cfg.summary_decay_factor = 0.5
    cfg.snapshot_decay_factor = 0.6
    
    # Mocking dependencies for EventService
    svc = EventService(cfg, None, None, None, None, None)
    
    raw_len = 1000
    budget = svc._compute_budget(raw_len)
    
    print(f"Raw Len: {raw_len}")
    print(f"Total Budget: {budget.total_budget}")
    print("\nSummary Budgets (L1->L10):")
    for lvl, b in budget.summary_level_budgets.items():
        print(f"  {lvl}: {b}")
        
    print("\nSnapshot Budgets (L3->L1):")
    for lvl, b in budget.snapshot_level_budgets.items():
        print(f"  {lvl}: {b}")

if __name__ == "__main__":
    test_utils()
    test_budgeting()
