import sys
import os
sys.path.insert(0, 'src')

from rems.config import REMSConfig, StorageConfig, UserMode, LLMConfig
from rems.pipeline import REMSPipeline
from rems.llm.provider import LLMProvider
from rems.storage.database import Database
from rems.storage.vector_store import VectorStore
import json

def test():
    config = REMSConfig(
        user_mode=UserMode.MULTI,
        storage=StorageConfig(database_url="sqlite:///:memory:"),
        llm=LLMConfig(provider="openai", model="gpt-4o", base_url="none", api_key="none")
    )
    
    pipeline = REMSPipeline.from_config(config)
    print(f"role_skill: {pipeline.role_skill}")
    
    raw_input = "贾宝玉走进大观园。"
    pipeline.ingest(raw_input)

if __name__ == "__main__":
    test()
