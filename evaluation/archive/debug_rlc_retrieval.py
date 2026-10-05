import sys
from pathlib import Path

# ------------------------------------------------------------
# Allow imports from project root
# ------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ------------------------------------------------------------
# UTF-8 output for mathematical symbols
# ------------------------------------------------------------

sys.stdout.reconfigure(
    encoding="utf-8"
)


# ------------------------------------------------------------
# RETRIEVERS TO DEBUG
# ------------------------------------------------------------

from src.retrieval import (
    wide_hybrid_retriever,
    wide_hybrid_rerank_retriever,
    retrieve_with_neighbor_context,
)


# ------------------------------------------------------------
# QUERY
# ------------------------------------------------------------

query = (
    "parallel RLC circuit current response "
    "limiting case limit as resistance "
    "R approaches infinity"
)


# ------------------------------------------------------------
# DISPLAY HELPER
# ------------------------------------------------------------

def show_docs(title, docs):

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    for i, doc in enumerate(
        docs,
        start=1,
    ):

        metadata = doc.metadata

        print(
            f"\n[{i}] "
            f"{metadata.get('source')} | "
            f"page={metadata.get('page')} | "
            f"chunk={metadata.get('chunk_index')}"
        )

        text = (
            doc.page_content
            .replace("\n", " ")
        )

        print(
            text[:900]
        )


# ------------------------------------------------------------
# 1. WIDE RAW HYBRID CANDIDATES
# ------------------------------------------------------------

raw_docs = (
    wide_hybrid_retriever.invoke(
        query
    )
)

show_docs(
    "WIDE RAW HYBRID",
    raw_docs,
)


# ------------------------------------------------------------
# 2. WIDE CROSS-ENCODER RERANKING
# ------------------------------------------------------------

reranked_docs = (
    wide_hybrid_rerank_retriever.invoke(
        query
    )
)

show_docs(
    "WIDE RERANKED TOP 5",
    reranked_docs,
)


# ------------------------------------------------------------
# 3. WIDE RETRIEVAL + PAGE EXPANSION
# ------------------------------------------------------------

expanded_docs = (
    retrieve_with_neighbor_context(
        query,
        page_radius=1,
        max_docs=9,
        use_wide=True,
    )
)

show_docs(
    "WIDE RERANKED + PAGE EXPANSION",
    expanded_docs,
)