import os
import json

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import (
    CrossEncoderReranker,
)

from src.config import (
    embeddings,
    cross_encoder,
    CHROMA_DIR,
    DATA_DIR,
)


# ============================================================
# DENSE RETRIEVER
# ============================================================

chroma_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="ultimate_docs",
)

dense_retriever = chroma_store.as_retriever(
    search_kwargs={"k": 5}
)


# ============================================================
# BM25 RETRIEVER
# ============================================================

bm25_data_path = os.path.join(
    DATA_DIR,
    "bm25_corpus.json",
)

if not os.path.exists(bm25_data_path):
    raise FileNotFoundError(
        "BM25 corpus not found. "
        "Run `python -m src.ingest` first."
    )

with open(
    bm25_data_path,
    "r",
    encoding="utf-8",
) as file:
    bm25_data = json.load(file)


bm25_documents = [
    Document(
        page_content=item["content"],
        metadata=item["metadata"],
    )
    for item in bm25_data
]


bm25_retriever = BM25Retriever.from_documents(
    bm25_documents
)

bm25_retriever.k = 5

# ============================================================
# HYBRID RETRIEVER
# ============================================================

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        dense_retriever,
    ],
    weights=[0.5, 0.5],
)


# ============================================================
# HYBRID + CROSS-ENCODER RERANKING
# ============================================================

reranker = CrossEncoderReranker(
    model=cross_encoder,
    top_n=3,
)

hybrid_rerank_retriever = (
    ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )
)


RETRIEVERS = {
    "dense": dense_retriever,
    "bm25": bm25_retriever,
    "hybrid": hybrid_retriever,
    "hybrid_rerank": hybrid_rerank_retriever,
}