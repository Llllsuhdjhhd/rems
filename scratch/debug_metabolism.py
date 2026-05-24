import sys
import os
import logging

# Ensure stdout is unbuffered
sys.stdout.reconfigure(line_buffering=True)

# Add src to path
sys.path.append(os.path.abspath("src"))

from rems.config import REMSConfig
from rems.services.metabolism_service import MetabolismService
from rems.services.event_service import EventService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.database import Database
from rems.storage.repository import EventRepository, RoleRepository, MetabolismRepository
from rems.storage.vector_store import VectorStore
from rems.llm.provider import LLMProvider

# Setup logging to console
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def run_debug_round():
    cfg = REMSConfig()
    # Ensure we use flash model to avoid quota issues
    cfg.llm.task_models.default = "qwen3.5-flash"
    cfg.llm.task_models.summary = "qwen3.5-flash"
    cfg.llm.task_models.boundary_detection = "qwen3.5-flash"
    cfg.llm.task_models.role_extraction = "qwen3.5-flash"
    
    db_path = "sqlite:///rems_debug.db"
    if os.path.exists("rems_debug.db"):
        os.remove("rems_debug.db")
        
    db = Database(db_path)
    db.create_tables()
    
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    meta_repo = MetabolismRepository(db)
    
    # We might have torch issues with VectorStore, let's see.
    # If it fails, I'll mock it.
    try:
        vector_store = VectorStore(cfg)
    except Exception as e:
        print(f"VectorStore failed (expected torch issue): {e}")
        from unittest.mock import MagicMock
        vector_store = MagicMock()
        vector_store.search.return_value = []
    
    llm = LLMProvider(cfg)
    summary_skill = SummaryGenerationSkill(llm, cfg)
    role_skill = RoleExtractionSkill(llm, cfg)
    event_svc = EventService(cfg, llm, event_repo, role_repo, summary_skill, role_skill)
    boundary_skill = BoundaryDetectionSkill(llm, cfg)
    
    metabolism = MetabolismService(cfg, meta_repo, boundary_skill, event_svc) 
    # Wait, MetabolismService might need RecallService?
    # No, it needs meta_repo.
    
    test_input = "甄士隐正在书房坐着，忽听外面喧嚷进来。原来是隔壁葫芦庙里失了火。"
    print(f"Processing input: {test_input}")
    
    result = metabolism.process_input(test_input)
    print("\nProcessing Result:")
    print(f"Events completed: {len(result.completed_events)}")
    for i, ev in enumerate(result.completed_events):
        print(f"Event {i+1} Content Raw: {ev.content_raw}")
        print(f"Event {i+1} Summaries: {list(ev.summaries.keys())}")
        for lvl, text in ev.summaries.items():
            print(f"  {lvl} ({len(text)} chars): {text}")

if __name__ == "__main__":
    run_debug_round()
