import csv
import json
import statistics
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from src.config import llm
from src.state import CitationVerification


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "verifier_thinking_vs_nothink.csv"
)


# ============================================================
# VERIFIER SYSTEM PROMPT
# Keep this synchronized with the production verifier prompt.
# ============================================================

VERIFIER_SYSTEM_PROMPT = """
You are a strict grounding and answer-completeness verifier.

You receive:
1. the user's exact question;
2. the retrieved context;
3. the generated answer.

Your job is NOT to rewrite or correct the answer.

Evaluate two separate things:

A. CORE ANSWER
The core answer is the minimum information required to answer
the user's actual question.

Set core_answer_supported=true ONLY when that required answer is
correct and explicitly supported by the retrieved context.

For mathematical questions, the core answer fails if any required
equation has a wrong or missing:
- coefficient
- variable
- factor
- sign
- exponent
- root
- derivative order
- subscript
- operator
- equality relationship

Example:

Context:
iR = (L/R) diL/dt

Answer:
iR = (1/R) diL/dt

core_answer_supported MUST be false because the required factor L
is missing.

LIMITING-RESPONSE COMPLETENESS:

When the question asks what a response approaches, becomes,
reduces to, or equals under a limiting condition, and the context
provides an explicit resulting expression, the complete explicit
expression is the required core answer.

Do NOT accept only one term, component, average value,
steady-state value, DC component, or qualitative interpretation
unless the question explicitly asks for that quantity.

Example:

Context:
As R approaches infinity, the solution reduces to

iL(t) = Is - Is cos(omega_0 t)

Question:
What does the current response approach as R becomes very large?

Answer:
iL = Is

This MUST be marked core_answer_supported=false because the
explicit limiting response contains the additional
-Is cos(omega_0 t) term.

B. OPTIONAL / EXTRA CLAIMS
An answer may contain the correct core answer but also contain
unnecessary unsupported material.

If an extra claim is unsupported while the required core answer
remains correct:

- set is_supported=false;
- keep core_answer_supported=true;
- put the unsupported material in unsupported_claims.

CRITICAL REDACTION RULE:

Every item in unsupported_claims MUST be copied EXACTLY and
VERBATIM from the generated answer.

Do not paraphrase.
Do not summarize.
Do not correct the text.
Return the smallest complete removable substring that can be
deleted without changing the supported core answer.

Include punctuation in the copied substring when appropriate.

If the required/core answer itself is incorrect or unsupported,
set core_answer_supported=false.

Set is_supported=true ONLY when every factual claim in the answer
is supported.

When is_supported=true:
- core_answer_supported must also be true;
- unsupported_claims must be empty.

Never generate a corrected answer.
"""


# ============================================================
# TWO IDENTICAL VERIFIERS
# Difference: /no_think only
# ============================================================

structured_verifier = llm.with_structured_output(
    CitationVerification,
    method="json_schema",
)


thinking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            VERIFIER_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Question:
{question}

Retrieved context:
{context}

Generated answer:
{generation}
""",
        ),
    ]
)


nothink_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            VERIFIER_SYSTEM_PROMPT,
        ),
        (
            "human",
            """
Question:
{question}

Retrieved context:
{context}

Generated answer:
{generation}

/no_think
""",
        ),
    ]
)


thinking_chain = (
    thinking_prompt
    | structured_verifier
)

nothink_chain = (
    nothink_prompt
    | structured_verifier
)


# ============================================================
# FIXED VERIFIER REGRESSION CASES
# ============================================================

PARALLEL_CURRENT_CONTEXT = """
[SOURCE 1]
Parallel RLC circuit derivation.

The resistor and capacitor currents are:

iR = (L/R) diL/dt

iC = LC d²iL/dt²
"""


LIMIT_CONTEXT = """
[SOURCE 1]
An important limiting case occurs as R approaches infinity.

Then alpha becomes very small compared with omega_0,
omega_d approaches omega_0, and e^(-alpha t) approaches 1.

The solution reduces to:

iL(t) = Is - Is cos(omega_0 t)
"""


TEST_CASES = [
    {
        "id": "supported_resistor",
        "question": (
            "In the parallel RLC derivation, how is the "
            "resistor current expressed in terms of the "
            "inductor current?"
        ),
        "context": PARALLEL_CURRENT_CONTEXT,
        "generation": (
            "The resistor current is "
            "iR = (L/R) diL/dt. [SOURCE 1]"
        ),
        "expected_is_supported": True,
        "expected_core_supported": True,
        "expected_redaction": None,
    },
    {
        "id": "missing_L",
        "question": (
            "In the parallel RLC derivation, how is the "
            "resistor current expressed in terms of the "
            "inductor current?"
        ),
        "context": PARALLEL_CURRENT_CONTEXT,
        "generation": (
            "The resistor current is "
            "iR = (1/R) diL/dt. [SOURCE 1]"
        ),
        "expected_is_supported": False,
        "expected_core_supported": False,
        "expected_redaction": None,
    },
    {
        "id": "wrong_sign",
        "question": (
            "In the parallel RLC derivation, how is the "
            "resistor current expressed in terms of the "
            "inductor current?"
        ),
        "context": PARALLEL_CURRENT_CONTEXT,
        "generation": (
            "The resistor current is "
            "iR = -(L/R) diL/dt. [SOURCE 1]"
        ),
        "expected_is_supported": False,
        "expected_core_supported": False,
        "expected_redaction": None,
    },
    {
        "id": "supported_limit",
        "question": (
            "For the parallel RLC circuit, what does the "
            "current response approach when resistance R "
            "becomes very large?"
        ),
        "context": LIMIT_CONTEXT,
        "generation": (
            "The response approaches "
            "iL(t) = Is - Is cos(omega_0 t). [SOURCE 1]"
        ),
        "expected_is_supported": True,
        "expected_core_supported": True,
        "expected_redaction": None,
    },
    {
        "id": "incomplete_limit",
        "question": (
            "For the parallel RLC circuit, what does the "
            "current response approach when resistance R "
            "becomes very large?"
        ),
        "context": LIMIT_CONTEXT,
        "generation": (
            "The current approaches iL = Is. [SOURCE 1]"
        ),
        "expected_is_supported": False,
        "expected_core_supported": False,
        "expected_redaction": None,
    },
    {
        "id": "correct_core_bad_extra",
        "question": (
            "For the parallel RLC circuit, what does the "
            "current response approach when resistance R "
            "becomes very large?"
        ),
        "context": LIMIT_CONTEXT,
        "generation": (
            "The response approaches "
            "iL(t) = Is - Is cos(omega_0 t). [SOURCE 1] "
            "The response is critically damped."
        ),
        "expected_is_supported": False,
        "expected_core_supported": True,
        "expected_redaction": (
            "The response is critically damped."
        ),
    },
]


# ============================================================
# RUN ONE CASE
# ============================================================

def run_case(
    mode,
    chain,
    case,
):

    start = time.perf_counter()

    result = chain.invoke(
        {
            "question":
                case["question"],
            "context":
                case["context"],
            "generation":
                case["generation"],
        }
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    decision_pass = (
        result.is_supported
        == case["expected_is_supported"]
        and
        result.core_answer_supported
        == case["expected_core_supported"]
    )

    expected_redaction = (
        case["expected_redaction"]
    )

    if expected_redaction is None:

        redaction_pass = True

    else:

        redaction_pass = (
            expected_redaction
            in result.unsupported_claims
        )

    overall_pass = (
        decision_pass
        and redaction_pass
    )

    return {
        "mode": mode,
        "case_id": case["id"],

        "expected_is_supported":
            case["expected_is_supported"],

        "actual_is_supported":
            result.is_supported,

        "expected_core_supported":
            case["expected_core_supported"],

        "actual_core_supported":
            result.core_answer_supported,

        "decision_pass":
            int(decision_pass),

        "redaction_pass":
            int(redaction_pass),

        "overall_pass":
            int(overall_pass),

        "latency_ms":
            latency_ms,

        "unsupported_claims":
            json.dumps(
                result.unsupported_claims,
                ensure_ascii=False,
            ),

        "reasoning":
            result.reasoning,
    }


# ============================================================
# WARM-UP
# ============================================================

print("\n🔥 Warming up local LLM...")

try:

    llm.invoke(
        "Reply with exactly: OK /no_think"
    )

    print("✅ Warm-up complete.")

except Exception as exc:

    print(
        f"⚠️ Warm-up failed: {exc}"
    )


# ============================================================
# BENCHMARK
# ============================================================

results = []

modes = [
    (
        "thinking",
        thinking_chain,
    ),
    (
        "no_think",
        nothink_chain,
    ),
]


for mode, chain in modes:

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"VERIFIER MODE: {mode}"
    )

    print(
        "=" * 72
    )

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print(
            f"\n[{index:02d}/{len(TEST_CASES)}] "
            f"{case['id']}"
        )

        try:

            row = run_case(
                mode,
                chain,
                case,
            )

            results.append(row)

            status = (
                "✅ PASS"
                if row["overall_pass"]
                else "❌ FAIL"
            )

            print(
                f"{status} | "
                f"supported="
                f"{row['actual_is_supported']} | "
                f"core="
                f"{row['actual_core_supported']} | "
                f"{row['latency_ms'] / 1000:.2f}s"
            )

        except Exception as exc:

            print(
                f"❌ ERROR: {exc}"
            )

            results.append(
                {
                    "mode": mode,
                    "case_id": case["id"],
                    "expected_is_supported":
                        case["expected_is_supported"],
                    "actual_is_supported": "",
                    "expected_core_supported":
                        case[
                            "expected_core_supported"
                        ],
                    "actual_core_supported": "",
                    "decision_pass": 0,
                    "redaction_pass": 0,
                    "overall_pass": 0,
                    "latency_ms": 0,
                    "unsupported_claims": "",
                    "reasoning": "",
                    "error": str(exc),
                }
            )


# ============================================================
# SAVE CSV
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
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "VERIFIER REGRESSION SUMMARY"
)

print(
    "=" * 72
)


for mode, _ in modes:

    mode_rows = [
        row
        for row in results
        if row["mode"] == mode
    ]

    passed = sum(
        row["overall_pass"]
        for row in mode_rows
    )

    latencies = [
        row["latency_ms"]
        for row in mode_rows
        if row["latency_ms"] > 0
    ]

    print(
        f"\n{mode}:"
    )

    print(
        f"  Correct: "
        f"{passed}/{len(mode_rows)}"
    )

    if latencies:

        print(
            f"  Mean latency: "
            f"{statistics.mean(latencies) / 1000:.2f}s"
        )

        print(
            f"  Median latency: "
            f"{statistics.median(latencies) / 1000:.2f}s"
        )


print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)