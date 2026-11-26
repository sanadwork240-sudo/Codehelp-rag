
"""Central settings. Change values here, not inside the other files."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load local .env values before reading settings.
load_dotenv()


def get_setting(name, default=None):
    """
    Read a setting from environment variables or Streamlit secrets.

    Environment variables take priority. Streamlit secrets are used
    when running on Streamlit Community Cloud.
    """
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st

        value = st.secrets.get(name, default)

        if value:
            return value

    except Exception:
        pass

    return default


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE = Path(__file__).parent

RAW_DIR = BASE / "data" / "raw"
CHROMA_DIR = BASE / "data" / "chroma_db"
DB_PATH = BASE / "data" / "helpdesk.db"


# --------------------------------------------------
# API KEYS AND ADMIN SETTINGS
# --------------------------------------------------

GEMINI_API_KEY = get_setting("GEMINI_API_KEY")
ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")


# --------------------------------------------------
# VECTOR DATABASE SETTINGS
# --------------------------------------------------

COLLECTION = "helpdesk"

EMBED_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------
# CHUNKING AND RETRIEVAL SETTINGS
# --------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TOP_K = 4

SIM_THRESHOLD = 0.35


# --------------------------------------------------
# GEMINI MODEL SETTINGS
# --------------------------------------------------

LLM_MODEL = get_setting("LLM_MODEL", "gemini-3.8-flash")

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]


# --------------------------------------------------
# CHATBOT SETTINGS
# --------------------------------------------------

NOT_FOUND_MSG = "I couldn't find this in the documents."

MAX_QUESTIONS_PER_SESSION = 15

MAX_QUESTION_CHARS = 500
