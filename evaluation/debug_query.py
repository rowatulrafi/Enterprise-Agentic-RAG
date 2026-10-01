from src.retrieval import (
    dense_retriever,
    bm25_retriever,
    hybrid_retriever,
    cross_encoder,
)


QUESTION = (
    "What happens to a series RLC circuit "
    "when resistance R approaches zero?"
)


def print_docs(name, docs):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for rank, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        preview = (
            doc.page_content
            .replace("\n", " ")
            [:250]
        )

        print(
            f"\nRank {rank}"
            f"\nSource: {source}"
            f"\nPage: {page}"
            f"\nText: {preview}"
        )


# ------------------------------------------------------------
# Base retrievers
# ------------------------------------------------------------

dense_docs = dense_retriever.invoke(
    QUESTION
)

bm25_docs = bm25_retriever.invoke(
    QUESTION
)

hybrid_docs = hybrid_retriever.invoke(
    QUESTION
)


print_docs(
    "DENSE",
    dense_docs
)

print_docs(
    "BM25",
    bm25_docs
)

print_docs(
    "HYBRID",
    hybrid_docs
)


# ------------------------------------------------------------
# Cross-encoder scores
# ------------------------------------------------------------

pairs = [
    (
        QUESTION,
        doc.page_content
    )
    for doc in hybrid_docs
]

scores = cross_encoder.score(
    pairs
)


scored_docs = list(
    zip(
        hybrid_docs,
        scores
    )
)

scored_docs.sort(
    key=lambda item: item[1],
    reverse=True
)


print("\n" + "=" * 80)
print("CROSS-ENCODER SCORES")
print("=" * 80)

for rank, (doc, score) in enumerate(
    scored_docs,
    start=1
):

    source = doc.metadata.get(
        "source",
        "unknown"
    )

    page = doc.metadata.get(
        "page",
        "unknown"
    )

    preview = (
        doc.page_content
        .replace("\n", " ")
        [:250]
    )

    print(
        f"\nRank {rank}"
        f"\nScore: {score:.4f}"
        f"\nSource: {source}"
        f"\nPage: {page}"
        f"\nText: {preview}"
    )