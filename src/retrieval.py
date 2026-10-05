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

# ============================================================
# LAYOUT-PRESERVED PARENT PAGE CONTEXT
# ============================================================

page_context_path = os.path.join(
    DATA_DIR,
    "page_context.json",
)

page_context_lookup = {}

if os.path.exists(
    page_context_path
):

    with open(
        page_context_path,
        "r",
        encoding="utf-8",
    ) as file:

        page_context_data = (
            json.load(file)
        )

    page_context_lookup = {
        (
            item["source"],
            item["page"],
        ): item
        for item in page_context_data
    }

def hydrate_parent_context(
    documents,
):
    """
    Replace retrieved child chunks with their
    layout-preserved parent-page representation.

    One page is included only once.
    """

    hydrated = []
    seen_pages = set()

    for doc in documents:

        source = doc.metadata.get(
            "source"
        )

        page = doc.metadata.get(
            "page"
        )

        key = (
            source,
            page,
        )

        if key in seen_pages:
            continue

        seen_pages.add(key)

        parent = (
            page_context_lookup.get(
                key
            )
        )

        if (
            parent
            and parent.get("content")
        ):

            metadata = dict(
                doc.metadata
            )

            metadata[
                "context_level"
            ] = "parent_page"

            metadata[
                "retrieval_excerpt"
            ] = doc.page_content

            hydrated.append(
                Document(
                    page_content=
                        parent["content"],
                    metadata=metadata,
                )
            )

        else:

            metadata = dict(
                doc.metadata
            )

            metadata[
                "retrieval_excerpt"
            ] = doc.page_content

            hydrated.append(
                Document(
                    page_content=
                        doc.page_content,
                    metadata=metadata,
                )
            )

    return hydrated

# ============================================================
# PAGE LOOKUP FOR CONTEXT EXPANSION
# ============================================================

page_lookup = {}

for doc in bm25_documents:

    metadata = doc.metadata

    key = (
        metadata.get("source"),
        metadata.get("page"),
    )

    if None in key:
        continue

    page_lookup.setdefault(
        key,
        []
    ).append(doc)


# Keep chunks on each page in their original order.
for docs in page_lookup.values():

    docs.sort(
        key=lambda doc:
            doc.metadata.get(
                "chunk_index",
                0,
            )
    )

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

# ============================================================
# WIDE FALLBACK RETRIEVER
# ============================================================

wide_dense_retriever = (
    chroma_store.as_retriever(
        search_kwargs={
            "k": 12
        }
    )
)


wide_bm25_retriever = (
    BM25Retriever.from_documents(
        bm25_documents
    )
)

wide_bm25_retriever.k = 12


wide_hybrid_retriever = (
    EnsembleRetriever(
        retrievers=[
            wide_bm25_retriever,
            wide_dense_retriever,
        ],
        weights=[
            0.5,
            0.5,
        ],
    )
)


wide_reranker = (
    CrossEncoderReranker(
        model=cross_encoder,
        top_n=5,
    )
)


wide_hybrid_rerank_retriever = (
    ContextualCompressionRetriever(
        base_compressor=wide_reranker,
        base_retriever=wide_hybrid_retriever,
    )
)

def retrieve_with_neighbor_context(
    query: str,
    page_radius: int = 1,
    max_docs: int = 7,
    use_wide: bool = False,
):
    """
    Retrieve high-value pages using child-chunk search,
    optionally widen recall, then provide layout-preserved
    parent-page context to the LLM.
    """

    if use_wide:

        reranked_docs = (
            wide_hybrid_rerank_retriever
            .invoke(query)
        )

        raw_docs = (
            wide_hybrid_retriever
            .invoke(query)
        )

        # Precision first, then recall insurance.
        seed_docs = (
            reranked_docs
            + raw_docs[:8]
        )

    else:

        seed_docs = (
            hybrid_rerank_retriever
            .invoke(query)
        )

    expanded_docs = []
    seen_pages = set()

    def add_doc(doc):

        source = doc.metadata.get(
            "source"
        )

        page = doc.metadata.get(
            "page"
        )

        identity = (
            source,
            page,
        )

        if identity in seen_pages:
            return

        seen_pages.add(
            identity
        )

        expanded_docs.append(
            doc
        )

    # ----------------------------------------
    # Preserve ranked seed pages first.
    # ----------------------------------------

    for doc in seed_docs:

        add_doc(doc)

        if (
            len(expanded_docs)
            >= max_docs
        ):

            return hydrate_parent_context(
                expanded_docs[:max_docs]
            )

    # ----------------------------------------
    # Add neighboring pages.
    # ----------------------------------------

    seed_pages = list(
        expanded_docs
    )

    for doc in seed_pages:

        source = doc.metadata.get(
            "source"
        )

        page = doc.metadata.get(
            "page"
        )

        if (
            source is None
            or page is None
        ):
            continue

        for distance in range(
            1,
            page_radius + 1,
        ):

            neighboring_pages = [
                page + distance,
                page - distance,
            ]

            for neighbor_page in (
                neighboring_pages
            ):

                neighbor_docs = (
                    page_lookup.get(
                        (
                            source,
                            neighbor_page,
                        ),
                        [],
                    )
                )

                for neighbor_doc in (
                    neighbor_docs
                ):

                    add_doc(
                        neighbor_doc
                    )

                    if (
                        len(expanded_docs)
                        >= max_docs
                    ):

                        return (
                            hydrate_parent_context(
                                expanded_docs[
                                    :max_docs
                                ]
                            )
                        )

    return hydrate_parent_context(
        expanded_docs[:max_docs]
    )

RETRIEVERS = {
    "dense": dense_retriever,
    "bm25": bm25_retriever,
    "hybrid": hybrid_retriever,
    "hybrid_rerank": hybrid_rerank_retriever,
}