import csv
import json
import time
from pathlib import Path

from src.graph import app
from src.telemetry import (
    reset_latency_trace,
    get_latency_trace,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "holdout20.json"
)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# REPRESENTATIVE BENCHMARK
# ============================================================


OUTPUT_PATH = (
    RESULTS_DIR
    / "holdout20_first_run.csv"
)





# ============================================================
# LOAD DATASET
# ============================================================

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8",
) as file:
    full_dataset = json.load(file)



from src.config import llm

dataset = full_dataset


# ============================================================
# HELPERS
# ============================================================

SAFE_FALLBACK_PHRASES = [
    "i don't know",
    "does not contain enough",
    "not contain enough",
    "cannot safely answer",
    "can't safely answer",
]


def is_safe_fallback(answer):
    if not answer:
        return False

    lower = answer.lower()

    return any(
        phrase in lower
        for phrase in SAFE_FALLBACK_PHRASES
    )


def has_source_citation(answer):
    if not answer:
        return False

    return "[SOURCE " in answer.upper()

# ============================================================
# RUN ONE QUERY THROUGH THE REAL GRAPH
# ============================================================

def run_query(item):

    reset_latency_trace()

    initial_state = {
        "question":
            item["question"],

        "search_query":
            item["question"],

        "documents":
            [],

        "generation":
            "",

        "verification_feedback":
            None,

        "rewrite_count":
            0,

        "verification_retries":
            0,
    }

    start = time.perf_counter()

    rewrite_count = 0
    retrieve_count = 0
    generation_count = 0
    verification_count = 0
    verifier_rejections = 0

    rewritten_queries = []

    generation_history = []
    
    first_draft = ""
    final_answer = ""

    final_state = dict(
        initial_state
    )

    # --------------------------------------------------------
    # STREAM THE ACTUAL LANGGRAPH
    # --------------------------------------------------------

    for update in app.stream(
        initial_state,
        config={
            "recursion_limit": 20
        },
        stream_mode="updates",
    ):

        for node_name, node_update in update.items():

            if not isinstance(
                node_update,
                dict,
            ):
                continue

            # Keep a local copy of the evolving state.
            final_state.update(
                node_update
            )

            # ----------------------------------------------
            # RETRIEVAL
            # ----------------------------------------------

            if node_name == "retrieve":

                retrieve_count += 1

            # ----------------------------------------------
            # QUERY REWRITE
            # ----------------------------------------------

            elif node_name == "rewrite":

                rewrite_count += 1

                new_query = (
                    node_update.get(
                        "search_query"
                    )
                )

                if new_query:
                    rewritten_queries.append(
                        new_query
                    )

            # ----------------------------------------------
            # GENERATION
            # ----------------------------------------------

            elif node_name == "generate":

                generation_count += 1

                generation = (
                    node_update.get(
                        "generation",
                        "",
                    )
                )

                if (
                    generation
                    and not first_draft
                ):
                    first_draft = (
                        generation
                    )

                if generation: 

                    generation_history.append(generation)

                    final_answer = (
                        generation
                    )

            # ----------------------------------------------
            # VERIFICATION
            # ----------------------------------------------

            elif node_name == "verify":

                verification_count += 1

                feedback = (
                    node_update.get(
                        "verification_feedback"
                    )
                )

                if feedback:
                    verifier_rejections += 1

                # Safety wipe may replace generation
                # directly inside verify_node.
                verified_generation = (
                    node_update.get(
                        "generation"
                    )
                )

                if (
                    verified_generation
                    and is_safe_fallback(verified_generation)
                ):
                    verifier_rejections += 1
                
                if verified_generation:
                    final_answer = (
                        verified_generation
                    )
    
    timings = get_latency_trace()

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    retrieval_ms = timings.get(
        "retrieval_ms", 0.0
    )

    grading_ms = timings.get(
        "grading_ms", 0.0
    )

    rewrite_ms = timings.get(
        "rewrite_ms", 0.0
    )

    generation_ms = timings.get(
        "generation_ms", 0.0
    )

    verification_ms = timings.get(
        "verification_ms", 0.0
    )

    accounted_stage_ms = (
        retrieval_ms
        + grading_ms
        + rewrite_ms
        + generation_ms
        + verification_ms
    )

    framework_overhead_ms = max(
        0.0,
        latency_ms - accounted_stage_ms,
    )

    # In case the graph's final state owns the answer.
    if final_state.get(
        "generation"
    ):
        final_answer = (
            final_state["generation"]
        )

    return {
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
            item["question"],

        "expected_source":
            item["expected_source"],

        "expected_pages":
            ",".join(
                str(page)
                for page
                in item["expected_pages"]
            ),

        "retrieve_calls":
            retrieve_count,

        "rewrite_calls":
            rewrite_count,

        "rewritten_queries":
            " || ".join(
                rewritten_queries
            ),

        "generation_calls":
            generation_count,

        "verification_calls":
            verification_count,

        "verifier_rejections":
            verifier_rejections,

        "draft_changed":
            int(
                bool(first_draft)
                and bool(final_answer)
                and first_draft.strip()
                != final_answer.strip()
            ),

        "safe_fallback":
            int(
                is_safe_fallback(
                    final_answer
                )
            ),

        "has_source_citation":
            int(
                has_source_citation(
                    final_answer
                )
            ),

        "first_draft":
            first_draft,

        "final_answer":
            final_answer,

        "retrieval_ms": retrieval_ms,
        "grading_ms": grading_ms,
        "rewrite_ms": rewrite_ms,
        "generation_ms": generation_ms,
        "verification_ms": verification_ms,
        "accounted_stage_ms": accounted_stage_ms,
        "framework_overhead_ms": framework_overhead_ms,
        "end_to_end_ms": latency_ms,

        "generation_history":
            " ||| GENERATION SPLIT ||| ".join(
                generation_history
            ),

    }


# ============================================================
# BENCHMARK
# ============================================================

results = []


print(
    "\n"
    + "=" * 78
)

print(
    "END-TO-END RAG + VERIFICATION BENCHMARK"
)

print(
    "=" * 78
)

print(
    f"Queries: {len(dataset)}"
)

print(
    "=" * 78
)

print("\n🔥 Warming up local LLM...")

try:
    llm.invoke(
        "Reply with exactly: OK"
    )
    print("✅ Warm-up complete.")

except Exception as exc:
    print(
        f"⚠️ Warm-up failed: {exc}"
    )


for index, item in enumerate(
    dataset,
    start=1,
):

    print(
        f"\n[{index:02d}/{len(dataset)}] "
        f"{item['id']} "
        f"[{item.get('difficulty', 'unknown')}]"
    )

    print(
        f"Question: "
        f"{item['question']}"
    )

    try:

        row = run_query(
            item
        )

        results.append(
            row
        )

        print(
            "Result: "
            f"rewrite={row['rewrite_calls']} | "
            f"generate={row['generation_calls']} | "
            f"verifier_reject="
            f"{row['verifier_rejections']} | "
            f"fallback={row['safe_fallback']} | "
            f"{row['end_to_end_ms'] / 1000:.1f}s"
        )

        print(
            f"Final: "
            f"{row['final_answer'][:300]}"
        )

    except Exception as exc:

        print(
            f"❌ ERROR: {exc}"
        )

        results.append(
            {
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
                    item["question"],

                "expected_source":
                    item["expected_source"],

                "expected_pages":
                    ",".join(
                        str(page)
                        for page
                        in item[
                            "expected_pages"
                        ]
                    ),

                "retrieve_calls":
                    0,

                "rewrite_calls":
                    0,

                "rewritten_queries":
                    "",

                "generation_calls":
                    0,

                "verification_calls":
                    0,

                "verifier_rejections":
                    0,

                "draft_changed":
                    0,

                "safe_fallback":
                    0,

                "has_source_citation":
                    0,

                "first_draft":
                    "",

                "final_answer":
                    "",

                "end_to_end_ms":
                    0,

                "error":
                    str(exc),
            }
        )


# ============================================================
# SAVE
# ============================================================

all_fields = []

for row in results:
    for key in row.keys():
        if key not in all_fields:
            all_fields.append(key)


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=all_fields,
    )

    writer.writeheader()
    writer.writerows(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

successful = [
    row
    for row in results
    if row.get(
        "final_answer"
    )
]


if successful:

    n = len(successful)

    rewrites = sum(
        row["rewrite_calls"]
        for row in successful
    )

    rejected = sum(
        row[
            "verifier_rejections"
        ]
        for row in successful
    )

    corrected = sum(
        row["draft_changed"]
        for row in successful
    )

    fallbacks = sum(
        row["safe_fallback"]
        for row in successful
    )

    citations = sum(
        row["has_source_citation"]
        for row in successful
    )

    avg_latency = (
        sum(
            row["end_to_end_ms"]
            for row in successful
        )
        / n
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        f"Completed              : "
        f"{n}/{len(dataset)}"
    )

    print(
        f"Query rewrites         : "
        f"{rewrites}"
    )

    print(
        f"Verifier rejections    : "
        f"{rejected}"
    )

    print(
        f"Answers changed        : "
        f"{corrected}"
    )

    print(
        f"Safe fallbacks         : "
        f"{fallbacks}"
    )

    print(
        f"Answers with citations : "
        f"{citations}/{n}"
    )

    print(
        f"Avg end-to-end latency : "
        f"{avg_latency / 1000:.1f}s"
    )


print(
    f"\n📄 Results: "
    f"{OUTPUT_PATH}"
)