import json
import time
import csv
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.config import llm
from src.state import RewrittenQuery
from src.retrieval import hybrid_rerank_retriever


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "rewrite_results.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as file:
    DATASET = json.load(file)


# ============================================================
# QUERY REWRITER
# ============================================================

structured_rewriter = (
    llm.with_structured_output(
        RewrittenQuery,
        method="json_schema",
    )
)


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert retrieval query optimizer.

Rewrite the user's question into a concise search query
that improves retrieval from a technical document database.

Rules:
- Preserve the original meaning.
- Preserve important entities, dates, quantities and qualifiers.
- Remove conversational filler.
- Include useful technical synonyms when appropriate.
- Convert spelled-out Greek letters to symbols when useful.
- Preserve both natural-language and symbolic forms for
  technical concepts when this improves retrieval.
- Do NOT answer the question.
- Do NOT add facts that are not implied by the question.

Examples:

Question:
How is the damped oscillation frequency related to the
natural frequency and damping rate?

Optimized query:
damped oscillation frequency ωd natural frequency ω0
damping rate α relation underdamped RLC

Question:
What differential equation describes the current response
of the parallel RLC circuit?

Optimized query:
parallel RLC current response iL differential equation
R C L current source

Return only the optimized query.
""",
        ),
        (
            "human",
            "Question:\n{question}",
        ),
    ]
)


question_rewriter = (
    rewrite_prompt
    | structured_rewriter
)


# ============================================================
# HELPERS
# ============================================================

def normalize_source(source):
    if source is None:
        return ""

    return str(source).replace(
        "\\",
        "/",
    ).split("/")[-1]


def is_relevant(
    doc,
    item,
):
    source = normalize_source(
        doc.metadata.get(
            "source"
        )
    )

    page = doc.metadata.get(
        "page"
    )

    return (
        source
        == item["expected_source"]
        and page
        in item["expected_pages"]
    )


def first_relevant_rank(
    docs,
    item,
):
    for rank, doc in enumerate(
        docs,
        start=1,
    ):
        if is_relevant(
            doc,
            item,
        ):
            return rank

    return None


def hit_at_k(
    docs,
    item,
    k,
):
    return int(
        any(
            is_relevant(
                doc,
                item,
            )
            for doc
            in docs[:k]
        )
    )


def reciprocal_rank(
    docs,
    item,
):
    rank = first_relevant_rank(
        docs,
        item,
    )

    if rank is None:
        return 0.0

    return 1.0 / rank


def rewrite_question(
    question,
):
    result = question_rewriter.invoke(
        {
            "question":
                question
        }
    )

    if isinstance(
        result,
        dict,
    ):
        return result["query"]

    return result.query


def retrieve(
    query,
):
    start = time.perf_counter()

    docs = (
        hybrid_rerank_retriever
        .invoke(
            query
        )
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return (
        docs,
        latency_ms,
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


print(
    "\n"
    + "=" * 72
)

print(
    "QUERY REWRITE RETRIEVAL EVALUATION"
)

print(
    "=" * 72
)

print(
    f"Queries: {len(DATASET)}"
)

print(
    "=" * 72
)


for index, item in enumerate(
    DATASET,
    start=1,
):

    original_question = (
        item["question"]
    )

    difficulty = (
        item.get(
            "difficulty",
            "unknown",
        )
    )

    # --------------------------------------------------------
    # ORIGINAL QUERY
    # --------------------------------------------------------

    original_docs, original_latency = (
        retrieve(
            original_question
        )
    )

    original_h1 = hit_at_k(
        original_docs,
        item,
        1,
    )

    original_h3 = hit_at_k(
        original_docs,
        item,
        3,
    )

    original_rank = (
        first_relevant_rank(
            original_docs,
            item,
        )
    )

    original_rr = (
        reciprocal_rank(
            original_docs,
            item,
        )
    )

    # --------------------------------------------------------
    # REWRITE
    # --------------------------------------------------------

    rewrite_start = (
        time.perf_counter()
    )

    rewritten_question = (
        rewrite_question(
            original_question
        )
    )

    rewrite_latency = (
        time.perf_counter()
        - rewrite_start
    ) * 1000

    # --------------------------------------------------------
    # REWRITTEN QUERY RETRIEVAL
    # --------------------------------------------------------

    rewritten_docs, rewritten_latency = (
        retrieve(
            rewritten_question
        )
    )

    rewritten_h1 = hit_at_k(
        rewritten_docs,
        item,
        1,
    )

    rewritten_h3 = hit_at_k(
        rewritten_docs,
        item,
        3,
    )

    rewritten_rank = (
        first_relevant_rank(
            rewritten_docs,
            item,
        )
    )

    rewritten_rr = (
        reciprocal_rank(
            rewritten_docs,
            item,
        )
    )

    # --------------------------------------------------------
    # CLASSIFY CHANGE
    # --------------------------------------------------------

    if (
        original_h3 == 0
        and rewritten_h3 == 1
    ):
        outcome = "recovered"

    elif (
        original_h3 == 1
        and rewritten_h3 == 0
    ):
        outcome = "degraded"

    elif rewritten_rr > original_rr:
        outcome = "improved_rank"

    elif rewritten_rr < original_rr:
        outcome = "worse_rank"

    else:
        outcome = "unchanged"

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append(
        {
            "question_id":
                item["id"],

            "difficulty":
                difficulty,

            "category":
                item.get(
                    "category",
                    "unknown",
                ),

            "original_question":
                original_question,

            "rewritten_question":
                rewritten_question,

            "original_hit_at_1":
                original_h1,

            "original_hit_at_3":
                original_h3,

            "original_first_rank":
                original_rank,

            "original_rr":
                original_rr,

            "rewritten_hit_at_1":
                rewritten_h1,

            "rewritten_hit_at_3":
                rewritten_h3,

            "rewritten_first_rank":
                rewritten_rank,

            "rewritten_rr":
                rewritten_rr,

            "rewrite_outcome":
                outcome,

            "original_retrieval_ms":
                original_latency,

            "rewrite_ms":
                rewrite_latency,

            "rewritten_retrieval_ms":
                rewritten_latency,
        }
    )

    original_rank_display = (
        original_rank
        if original_rank
        is not None
        else "-"
    )

    rewritten_rank_display = (
        rewritten_rank
        if rewritten_rank
        is not None
        else "-"
    )

    marker = {
        "recovered": "🟢",
        "improved_rank": "⬆️",
        "degraded": "🔴",
        "worse_rank": "⬇️",
        "unchanged": "➖",
    }[outcome]

    print(
        f"{marker} "
        f"[{index:02d}/{len(DATASET)}] "
        f"{item['id']} "
        f"[{difficulty}] "
        f"| H3 "
        f"{original_h3}→"
        f"{rewritten_h3} "
        f"| Rank "
        f"{original_rank_display}→"
        f"{rewritten_rank_display} "
        f"| {outcome}"
    )

    print(
        f"   Rewrite: "
        f"{rewritten_question}"
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(
            results[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

n = len(results)


def avg(field):
    return (
        sum(
            row[field]
            for row
            in results
        )
        / n
    )


original_h1 = avg(
    "original_hit_at_1"
)

original_h3 = avg(
    "original_hit_at_3"
)

original_mrr = avg(
    "original_rr"
)

rewritten_h1 = avg(
    "rewritten_hit_at_1"
)

rewritten_h3 = avg(
    "rewritten_hit_at_3"
)

rewritten_mrr = avg(
    "rewritten_rr"
)


recovered = sum(
    row["rewrite_outcome"]
    == "recovered"
    for row in results
)

degraded = sum(
    row["rewrite_outcome"]
    == "degraded"
    for row in results
)

improved_rank = sum(
    row["rewrite_outcome"]
    == "improved_rank"
    for row in results
)

worse_rank = sum(
    row["rewrite_outcome"]
    == "worse_rank"
    for row in results
)


print(
    "\n"
    + "=" * 72
)

print(
    "SUMMARY"
)

print(
    "=" * 72
)

print(
    f"{'Metric':<20}"
    f"{'Original':>12}"
    f"{'Rewritten':>12}"
)

print(
    "-" * 44
)

print(
    f"{'Hit@1':<20}"
    f"{original_h1:>12.3f}"
    f"{rewritten_h1:>12.3f}"
)

print(
    f"{'Hit@3':<20}"
    f"{original_h3:>12.3f}"
    f"{rewritten_h3:>12.3f}"
)

print(
    f"{'MRR':<20}"
    f"{original_mrr:>12.3f}"
    f"{rewritten_mrr:>12.3f}"
)

print(
    "\nRewrite outcomes:"
)

print(
    f"Recovered failures : "
    f"{recovered}"
)

print(
    f"Improved rank      : "
    f"{improved_rank}"
)

print(
    f"Degraded H@3       : "
    f"{degraded}"
)

print(
    f"Worse rank         : "
    f"{worse_rank}"
)

print(
    f"\n📄 Results: "
    f"{OUTPUT_PATH}"
)