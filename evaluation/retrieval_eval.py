import json
import time
import csv
from pathlib import Path

from src.retrieval import RETRIEVERS


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "retrieval_results.csv"
SUMMARY_CSV = RESULTS_DIR / "retrieval_summary.csv"


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as file:
    DATASET = json.load(file)


# ============================================================
# HELPERS
# ============================================================

def first_relevant_rank(docs, item):

    for rank, doc in enumerate(
        docs,
        start=1,
    ):
        if is_relevant(doc, item):
            return rank

    return None

def normalize_source(source: str) -> str:
    """
    Normalize filenames for robust comparison.
    """

    return str(source).strip().lower()


def is_relevant(doc, item):
    """
    A retrieval hit is relevant when:

    1. source matches expected_source
    AND
    2. retrieved page is one of expected_pages
    """

    retrieved_source = normalize_source(
        doc.metadata.get("source", "")
    )

    retrieved_page = doc.metadata.get(
        "page"
    )

    expected_source = normalize_source(
        item["expected_source"]
    )

    expected_pages = item.get(
        "expected_pages",
        []
    )

    return (
        retrieved_source == expected_source
        and retrieved_page in expected_pages
    )


def reciprocal_rank(docs, item):
    """
    Reciprocal rank of first relevant result.

    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.333...
    No relevant result -> 0
    """

    for rank, doc in enumerate(
        docs,
        start=1,
    ):

        if is_relevant(doc, item):
            return 1.0 / rank

    return 0.0


def hit_at_k(docs, item, k):
    """
    1 if at least one relevant document occurs
    within top-k results.
    """

    return int(
        any(
            is_relevant(doc, item)
            for doc in docs[:k]
        )
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    detailed_results = []

    print("\n" + "=" * 70)
    print("RETRIEVAL BENCHMARK")
    print("=" * 70)

    print(
        f"Queries    : {len(DATASET)}"
    )

    print(
        f"Retrievers: {len(RETRIEVERS)}"
    )

    print("=" * 70)

    for retriever_name, retriever in RETRIEVERS.items():

        print(
            f"\n🔎 Evaluating: {retriever_name}"
        )

        for index, item in enumerate(
            DATASET,
            start=1,
        ):

            question = item["question"]

            start_time = time.perf_counter()

            docs = retriever.invoke(
                question
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            hit_1 = hit_at_k(
                docs,
                item,
                1,
            )

            hit_3 = hit_at_k(
                docs,
                item,
                3,
            )
            
            hit_5 = hit_at_k(
                docs,
                item,
                5,
            )

            first_rank = first_relevant_rank(
                docs,
                item,
            )

            rr = reciprocal_rank(
                docs,
                item,
            )

            top_results = []

            for rank, doc in enumerate(
                docs[:5],
                start=1,
            ):

                top_results.append({
                    "rank": rank,
                    "source":
                        doc.metadata.get(
                            "source"
                        ),
                    "page":
                        doc.metadata.get(
                            "page"
                        ),
                    "relevant":
                        is_relevant(
                            doc,
                            item,
                        ),
                })

            detailed_results.append({
                "retriever":
                    retriever_name,

                "question_id":
                    item["id"],

                "category":
                    item.get(
                        "category",
                        "unknown",
                    ),

                "difficulty":
                    item.get(
                        "difficulty",
                        "unknown",
                    ),

                "question":
                    question,

                "expected_source":
                    item[
                        "expected_source"
                    ],

                "expected_pages":
                    ",".join(
                        map(
                            str,
                            item[
                                "expected_pages"
                            ],
                        )
                    ),

                "hit_at_1":
                    hit_1,

                "hit_at_3":
                    hit_3,
                
                "hit_at_5":
                    hit_5,

                "first_relevant_rank":
                    first_rank,

                "reciprocal_rank":
                    rr,

                "latency_ms":
                    latency_ms,

                "retrieved":
                    json.dumps(
                        top_results,
                        ensure_ascii=False,
                    ),
            })

            status = (
                "✅"
                if hit_3
                else "❌"
            )

            rank_display = (
                first_rank
                if first_rank is not None
                else "-"
            )

            difficulty = item.get(
                "difficulty",
                "unknown",
            )

            print(
                f"{status} "
                f"[{index:02d}/{len(DATASET)}] "
                f"{item['id']} "
                f"[{difficulty}] "
                f"| H@1={hit_1} "
                f"| H@3={hit_3} "
                f"| H@5={hit_5} "
                f"| First={rank_display} "
                f"| RR={rr:.3f} "
                f"| {latency_ms:.1f}ms"
            )

    save_detailed_results(
        detailed_results
    )

    build_summary(
        detailed_results
    )


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

def save_detailed_results(results):

    fieldnames = list(
        results[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\n📄 Detailed results: "
        f"{OUTPUT_CSV}"
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(results):

    summary_rows = []

    for retriever_name in RETRIEVERS:

        rows = [
            row
            for row in results
            if row["retriever"]
            == retriever_name
        ]

        n = len(rows)

        hit_1 = sum(
            row["hit_at_1"]
            for row in rows
        ) / n

        hit_3 = sum(
            row["hit_at_3"]
            for row in rows
        ) / n

        hit_5 = sum(
            row["hit_at_5"]
            for row in rows
        ) / n       

        mrr = sum(
            row["reciprocal_rank"]
            for row in rows
        ) / n

        avg_latency = sum(
            row["latency_ms"]
            for row in rows
        ) / n

        summary_rows.append({
            "retriever":
                retriever_name,

            "queries":
                n,

            "hit_at_1":
                round(hit_1, 4),

            "hit_at_3":
                round(hit_3, 4),
            
            "hit_at_5":
                round(hit_5, 4),

            "mrr":
                round(mrr, 4),

            "avg_latency_ms":
                round(
                    avg_latency,
                    2,
                ),

        })

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                summary_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print(
        f"📊 Summary: "
        f"{SUMMARY_CSV}"
    )

    print("\n" + "=" * 70)

    print(
        f"{'Retriever':<20}"
        f"{'H@1':>10}"
        f"{'H@3':>10}"
        f"{'MRR':>10}"
        f"{'Latency':>15}"
    )

    print("-" * 70)

    for row in summary_rows:

        print(
            f"{row['retriever']:<20}"
            f"{row['hit_at_1']:>10.3f}"
            f"{row['hit_at_3']:>10.3f}"
            f"{row['mrr']:>10.3f}"
            f"{row['avg_latency_ms']:>12.1f} ms"
        )


if __name__ == "__main__":
    evaluate()