import os
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

load_dotenv()

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
CLEAN_JSON_DIR = os.path.join(DATA_DIR, "clean_json")
REVIEW_DIR = os.path.join(DATA_DIR, "human_review_queue")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# --- HARDWARE OPTIMIZATION ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# --- MODELS ---
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://127.0.0.1:1234/v1"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen/qwen3-8b"
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    "lm-studio"
)

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    model_kwargs={"device": device}
)