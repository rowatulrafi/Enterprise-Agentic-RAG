import csv
import json
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.config import llm
from src.state import GradeDocuments, RewrittenQuery
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
    / "agentic_retrieval_results.csv"
)


# ============================================================
# DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as file:
    DATASET = json.load(file)


# ============================================================
# RELEVANCE GRADER
# ============================================================

structured_grader = (
    llm.with_structured_output(
        GradeDocuments,
        method="json_schema",
    )
)


grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval sufficiency grader.

You are given the user's question and the complete set of
retrieved passages that would be supplied to the answer generator.

Determine whether these passages, taken TOGETHER, contain enough
EXPLICIT evidence to answer the exact question faithfully.

Grade YES only when:
- the requested fact, relationship, formula, definition,
  comparison, or explanation is explicitly supported;
- the answer can be produced without guessing;
- the answer does not require deriving a missing relationship
  from loosely related information;
- the passages contain enough information to preserve important
  qualifiers, entities, dates, quantities, and scope.

Grade NO when:
- the passages are only about the same broad topic;
- they contain related variables but not the requested relation;
- they contain an example but not the requested general rule;
- a formula/equation is requested but the needed formula/equation
  itself is absent;
- answering would require outside knowledge or inference.

For mathematical questions:
- imperfect formatting is acceptable;
- symbols and equations count as evidence;
- but the requested mathematical relationship must actually
  appear in the retrieved context.

Do not answer the question.

Return:
binary_score = "yes" or "no"
"""
        ),
        (
            "human",
            """
Question:
{question}

Retrieved context:
{document}
"""
        ),
    ]
)


retrieval_grader = (
    grade_prompt
    | structured_grader
)


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
You optimize search queries for document retrieval.

Rewrite the user's question into a concise search query.

STRICT RULES:
- Preserve the exact intent.
- Preserve entities, dates, quantities and qualifiers.
- Remove conversational filler.
- You may add direct synonyms or symbolic equivalents only
  when they represent the SAME concept.
- Do not introduce related theories, frameworks, methods,
  applications, causes, or terminology not implied by the
  original question.
- For mathematical questions, retain natural-language terms
  while optionally adding their common symbols.
- Do not answer the question.

Examples:

"damping rate alpha"
→ "damping rate alpha α"

"damped oscillation frequency"
→ "damped oscillation frequency ωd"

Do NOT transform:
"nature connection"
into unrelated expansions such as
"biophilia hypothesis", "urban planning",
or "ecological identity".

Return only the optimized search query.
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

    return (
        str(source)
        .replace("\\", "/")
        .split("/")[-1]
    )


def is_relevant(doc, item):

    source = normalize_source(
        doc.metadata.get("source")
    )

    page = doc.metadata.get("page")

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


def retrieve(query):

    start = time.perf_counter()

    docs = (
        hybrid_rerank_retriever
        .invoke(query)
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return docs, latency_ms


def should_rewrite(
    question,
    docs,
):
    start = time.perf_counter()

    context_parts = []

    for index, doc in enumerate(
        docs,
        start=1,
    ):
        context_parts.append(
            f"[DOCUMENT {index}]\n"
            f"{doc.page_content}"
        )

    combined_context = (
        "\n\n".join(
            context_parts
        )
    )

    result = retrieval_grader.invoke(
        {
            "question":
                question,

            "document":
                combined_context,
        }
    )

    score = (
        result.binary_score
        .strip()
        .lower()
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    rewrite_needed = (
        score != "yes"
    )

    return (
        rewrite_needed,
        [score],
        latency_ms,
    )


def rewrite_question(question):

    start = time.perf_counter()

    result = (
        question_rewriter.invoke(
            {
                "question":
                    question
            }
        )
    )

    if isinstance(
        result,
        dict,
    ):
        rewritten = result["query"]
    else:
        rewritten = result.query

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    return (
        rewritten,
        latency_ms,
    )


# ============================================================
# EVALUATION
# ============================================================

results = []


print(
    "\n"
    + "=" * 76
)

print(
    "CONDITIONAL AGENTIC RETRIEVAL EVALUATION"
)

print(
    "=" * 76
)

print(
    f"Queries: {len(DATASET)}"
)

print(
    "=" * 76
)


for index, item in enumerate(
    DATASET,
    start=1,
):

    question = item["question"]

    # --------------------------------------------------------
    # ORIGINAL RETRIEVAL
    # --------------------------------------------------------

    original_docs, original_ms = (
        retrieve(question)
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
    # RELEVANCE GRADING
    # --------------------------------------------------------

    (
        rewrite_needed,
        grades,
        grading_ms,
    ) = should_rewrite(
        question,
        original_docs,
    )

    rewritten_question = None
    rewrite_ms = 0.0
    second_retrieval_ms = 0.0

    final_docs = original_docs

    # --------------------------------------------------------
    # CONDITIONAL REWRITE
    # --------------------------------------------------------

    if rewrite_needed:

        (
            rewritten_question,
            rewrite_ms,
        ) = rewrite_question(
            question
        )

        (
            final_docs,
            second_retrieval_ms,
        ) = retrieve(
            rewritten_question
        )

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    final_h1 = hit_at_k(
        final_docs,
        item,
        1,
    )

    final_h3 = hit_at_k(
        final_docs,
        item,
        3,
    )

    final_rank = (
        first_relevant_rank(
            final_docs,
            item,
        )
    )

    final_rr = (
        reciprocal_rank(
            final_docs,
            item,
        )
    )

    # --------------------------------------------------------
    # GRADER DIAGNOSTICS
    # --------------------------------------------------------

    # Ground truth says original retrieval worked,
    # but grader unnecessarily requested rewrite.
    grader_false_negative = int(
        original_h3 == 1
        and rewrite_needed
    )

    # Ground truth says original retrieval failed,
    # and grader correctly requested rewrite.
    grader_caught_failure = int(
        original_h3 == 0
        and rewrite_needed
    )

    if (
        original_h3 == 0
        and final_h3 == 1
    ):
        outcome = "recovered"

    elif (
        original_h3 == 1
        and final_h3 == 0
    ):
        outcome = "degraded"

    elif final_rr > original_rr:
        outcome = "improved_rank"

    elif final_rr < original_rr:
        outcome = "worse_rank"

    else:
        outcome = "unchanged"

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    row = {
        "question_id":
            item["id"],

        "difficulty":
            item.get(
                "difficulty",
                "unknown",
            ),

        "category":
            item.get(
                "category",
                "unknown",
            ),

        "question":
            question,

        "rewrite_triggered":
            int(
                rewrite_needed
            ),

        "grader_scores":
            "|".join(
                grades
            ),

        "grader_false_negative":
            grader_false_negative,

        "grader_caught_failure":
            grader_caught_failure,

        "rewritten_question":
            (
                rewritten_question
                or ""
            ),

        "original_hit_at_1":
            original_h1,

        "original_hit_at_3":
            original_h3,

        "original_first_rank":
            original_rank,

        "original_rr":
            original_rr,

        "final_hit_at_1":
            final_h1,

        "final_hit_at_3":
            final_h3,

        "final_first_rank":
            final_rank,

        "final_rr":
            final_rr,

        "outcome":
            outcome,

        "original_retrieval_ms":
            original_ms,

        "grading_ms":
            grading_ms,

        "rewrite_ms":
            rewrite_ms,

        "second_retrieval_ms":
            second_retrieval_ms,
    }

    results.append(row)

    original_rank_display = (
        original_rank
        if original_rank
        is not None
        else "-"
    )

    final_rank_display = (
        final_rank
        if final_rank
        is not None
        else "-"
    )

    marker = {
        "recovered": "🟢",
        "degraded": "🔴",
        "improved_rank": "⬆️",
        "worse_rank": "⬇️",
        "unchanged": "➖",
    }[outcome]

    route = (
        "REWRITE"
        if rewrite_needed
        else "KEEP"
    )

    print(
        f"{marker} "
        f"[{index:02d}/{len(DATASET)}] "
        f"{item['id']} "
        f"| route={route:<7} "
        f"| H3 "
        f"{original_h3}→{final_h3} "
        f"| rank "
        f"{original_rank_display}"
        f"→{final_rank_display} "
        f"| {outcome}"
    )

    if rewritten_question:

        print(
            f"   Rewrite: "
            f"{rewritten_question}"
        )


# ============================================================
# SAVE
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
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

n = len(results)


def average(field):

    return (
        sum(
            row[field]
            for row
            in results
        )
        / n
    )


original_h1 = average(
    "original_hit_at_1"
)

original_h3 = average(
    "original_hit_at_3"
)

original_mrr = average(
    "original_rr"
)

final_h1 = average(
    "final_hit_at_1"
)

final_h3 = average(
    "final_hit_at_3"
)

final_mrr = average(
    "final_rr"
)


rewrite_count = sum(
    row["rewrite_triggered"]
    for row in results
)

false_negatives = sum(
    row["grader_false_negative"]
    for row in results
)

caught_failures = sum(
    row["grader_caught_failure"]
    for row in results
)

raw_failures = sum(
    row["original_hit_at_3"] == 0
    for row in results
)

recovered = sum(
    row["outcome"]
    == "recovered"
    for row in results
)

degraded = sum(
    row["outcome"]
    == "degraded"
    for row in results
)


print(
    "\n"
    + "=" * 76
)

print(
    "SUMMARY"
)

print(
    "=" * 76
)

print(
    f"{'Metric':<24}"
    f"{'Raw':>10}"
    f"{'Agentic':>12}"
)

print(
    "-" * 46
)

print(
    f"{'Hit@1':<24}"
    f"{original_h1:>10.3f}"
    f"{final_h1:>12.3f}"
)

print(
    f"{'Hit@3':<24}"
    f"{original_h3:>10.3f}"
    f"{final_h3:>12.3f}"
)

print(
    f"{'MRR':<24}"
    f"{original_mrr:>10.3f}"
    f"{final_mrr:>12.3f}"
)

print(
    "\nRouting:"
)

print(
    f"Rewrites triggered      : "
    f"{rewrite_count}/{n}"
)

print(
    f"Raw H@3 failures        : "
    f"{raw_failures}"
)

print(
    f"Failures caught by grader: "
    f"{caught_failures}"
)

print(
    f"Grader false negatives : "
    f"{false_negatives}"
)

print(
    f"Failures recovered      : "
    f"{recovered}"
)

print(
    f"Successful queries lost : "
    f"{degraded}"
)

print(
    "\nLatency:"
)

print(
    f"Avg initial retrieval   : "
    f"{average('original_retrieval_ms'):.1f} ms"
)

print(
    f"Avg grading             : "
    f"{average('grading_ms'):.1f} ms"
)

print(
    f"Avg rewrite             : "
    f"{average('rewrite_ms'):.1f} ms"
)

print(
    f"Avg second retrieval    : "
    f"{average('second_retrieval_ms'):.1f} ms"
)

print(
    f"\n📄 Results: "
    f"{OUTPUT_PATH}"
)