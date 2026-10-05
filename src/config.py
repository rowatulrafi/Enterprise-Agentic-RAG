import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder,
)


load_dotenv()

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
CLEAN_JSON_DIR = DATA_DIR / "clean_json"
REVIEW_DIR = DATA_DIR / "human_review_queue"
CHROMA_DIR = BASE_DIR / "chroma_db"


# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"

elif torch.backends.mps.is_available():
    device = "mps"

else:
    device = "cpu"


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://127.0.0.1:1234/v1",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen/qwen3-8b",
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    "lm-studio",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": device,
    },
    encode_kwargs={
        "normalize_embeddings": True,
    },
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name=RERANKER_MODEL,
    model_kwargs={
        "device": device,
    },
)