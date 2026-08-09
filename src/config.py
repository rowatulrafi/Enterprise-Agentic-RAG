import os
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
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
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- MODELS ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    model_kwargs={"device": device}
)