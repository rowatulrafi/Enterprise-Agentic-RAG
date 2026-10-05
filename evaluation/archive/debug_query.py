import json
import sys
from pathlib import Path

from src.retrieval import (
    dense_retriever,
    bm25_retriever,
    hybrid_retriever,
    cross_encoder,
)

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "dataset.json"
)

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as file:
    dataset = json.load(file)


question_id = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "rlc_008"
)


item = next(
    x
    for x in dataset
    if x["id"] == question_id
)


QUESTION = item["question"]

print(f"\nQuestion ID : {question_id}")
print(f"Question    : {QUESTION}")
print(
    f"Expected    : "
    f"{item['expected_source']} "
    f"{item['expected_pages']}"
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