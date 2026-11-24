"""Central settings. Change values here, not inside the other files."""
from pathlib import Path

BASE = Path(__file__).parent
RAW_DIR = BASE / "data" / "raw"            # your PDFs / DOCX / TXT live here
CHROMA_DIR = BASE / "data" / "chroma_db"   # vector database (auto-created)
DB_PATH = BASE / "data" / "helpdesk.db"    # SQLite: tickets, feedback, chat logs

COLLECTION = "helpdesk"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800        # characters per chunk (try 500 vs 1000 in your evaluation)
CHUNK_OVERLAP = 120
TOP_K = 4               # chunks sent to the LLM (try 3 vs 5)
SIM_THRESHOLD = 0.30    # cosine similarity; below this the bot refuses + makes a ticket
LLM_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
NOT_FOUND_MSG = "I couldn't find this in the documents."
