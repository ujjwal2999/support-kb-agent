"""
Configuration settings for the knowledge base agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "models/gemini-2.5-flash"

# project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = PROJECT_ROOT / "index"

# chunking config - tried different values, 500/50 works best
CHUNK_SIZE = 500
CHUNK_OVERLAP =50

# retrieval
DEFAULT_K = 3
