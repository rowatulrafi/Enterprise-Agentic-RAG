import os
import json
import re

from src.telemetry import measure_stage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from src.config import llm, embeddings, cross_encoder, CHROMA_DIR, DATA_DIR
from src.state import GradeDocuments, RewrittenQuery, CitationVerification
from src.retrieval import (
    retrieve_with_neighbor_context,
)
# ==========================================
# 0. Helper for formatting context
# ==========================================
def format_context(
    documents,
    selected_indices=None,
):

    formatted_sections = []

    for index, doc in enumerate(
        documents,
        start=1,
    ):

        if (
            selected_indices is not None
            and index not in selected_indices
        ):
            continue

        formatted_sections.append(
            f"""
[SOURCE {index}]
Document: {doc["source"]}
Page: {doc["page"]}
Extraction: {doc["extraction_method"]}

{doc["content"]}
""".strip()
        )

    return "\n\n".join(
        formatted_sections
    )

def extract_cited_source_indices(
    generation: str,
):

    matches = re.findall(
        r"\[SOURCE\s+(\d+)\]",
        generation,
        flags=re.IGNORECASE,
    )

    return {
        int(match)
        for match in matches
    }

def redact_unsupported_claims(
    generation: str,
    unsupported_claims,
):
    """
    Delete verifier-identified unsupported spans.

    Safety rule:
    every span must occur verbatim in the generated answer.
    If even one span cannot be matched exactly, fail closed.
    """

    if not unsupported_claims:
        return None

    cleaned = generation

    for claim in unsupported_claims:

        claim = claim.strip()

        if not claim:
            return None

        # Verifier MUST provide an exact substring.
        if claim not in cleaned:
            print(
                "⚠️ Redaction failed: verifier returned "
                "a non-verbatim unsupported span."
            )
            return None

        cleaned = cleaned.replace(
            claim,
            "",
            1,
        )

    # Cosmetic cleanup only.
    # No factual rewriting occurs here.
    cleaned = re.sub(
        r"[ \t]+\n",
        "\n",
        cleaned,
    )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    cleaned = re.sub(
        r"[ \t]{2,}",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        cleaned,
    )

    cleaned = cleaned.strip()

    if not cleaned:
        return None

    return cleaned

# ==========================================
# 1. INITIALIZE HYBRID RETRIEVAL
# ==========================================

# ==========================================
# 2. INITIALIZE LLM CHAINS
# ==========================================
# Grader (Agentic Document Relevance)
structured_grader = llm.with_structured_output(
    GradeDocuments,
    method="json_schema"
)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval sufficiency grader.

You receive a user's question and the retrieval evidence excerpts that selected the pages available to the answer generator.

Return YES only if the retrieved context, taken together, contains enough explicit evidence to answer the exact question faithfully.

Return NO when:
- passages are merely about the same topic;
- requested numbers, qualifiers or relationships are absent;
- related variables are present but the requested relation is absent;
- an equation is requested but that equation is absent;
- answering requires outside knowledge or inference;
- an example is present but the requested general rule is absent.
- the question asks for an outcome under a condition, limit,
  threshold, or parameter regime, but the context does not
  explicitly connect that SAME condition to the requested outcome;
- a general equation, steady-state value, parameter definition,
  or nearby operating case is present, but the requested limiting
  or conditional result itself is absent.

For mathematical text, imperfect extraction is acceptable.
Equations and symbols count as evidence when they actually support the requested relationship.

Do not answer the question.
"""
        ),
        (
            "human",
            """
Question:
{question}

Retrieved evidence excerpts:
{document}
"""
        ),
    ]
)
retrieval_grader = grade_prompt | structured_grader

# Rewriter (Agentic Query Optimization)
structured_rewriter = llm.with_structured_output(
    RewrittenQuery,
    method="json_schema"
)
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You optimize search queries for document retrieval.

Preserve the user's exact intent.

Rules:
- Preserve entities, dates, numbers and qualifiers.
- Remove conversational filler.
- Add only direct synonyms or symbolic equivalents of
  concepts already present in the question.
- For mathematical concepts, retain the natural-language
  wording and optionally add common symbols.
- Do not introduce related theories, frameworks, causes,
  applications or terminology not implied by the question.
- Do not answer the question.

Examples:

"damping rate alpha"
-> "damping rate alpha α"

"damped oscillation frequency"
-> "damped oscillation frequency ωd"

Do not turn "nature connection" into unrelated expansions
such as "biophilia", "ecological identity", or
"environmental attachment".

- For limiting-language queries, preserve the original wording
  and add direct mathematical equivalents when applicable.

- Also add the retrieval terms "limit" and "limiting case"
  when the question explicitly asks what happens as a parameter
  becomes very large, very small, approaches zero, or approaches
  infinity.

Examples:
"becomes very large"
-> "becomes very large, approaches infinity, limit, limiting case"

"approaches zero"
-> "approaches zero, limit, limiting case"

- Never remove the original wording.

"""
        ),
        (
            "human",
            "Question:\n{question}",
        ),
    ]
)

question_rewriter = rewrite_prompt | structured_rewriter

# Generator (Drafting the Answer)
gen_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a grounded retrieval assistant.

Answer ONLY using the provided context.

ANSWERING AND GROUNDING RULES:

1. Answer ONLY using the retrieved context.

2. Answer only what the user actually asked.
   Give the shortest complete answer possible.
   Once every requested part is answered, STOP.

3. Every factual claim must be explicitly supported by the context.
   Do not use outside knowledge, even in examples, clarifications,
   parentheses, or common formulas.

4. Preserve exact entity, year, date, quantity, unit, metric,
   methodology, comparison, and qualifier relationships.
   Never transfer a qualifier or value to another entity, year,
   metric, or methodology.

5. If the source contains multiple versions of a metric,
   keep each value strictly paired with its correct definition
   and methodology.

6. For mathematical expressions, preserve the exact equation
   structure from the source. Do not alter exponents, roots,
   signs, subscripts, operators, or equality relationships.

7. Distinguish general rules from document-specific examples.
   Never present example-specific parameter values as universal rules.

8. Use causal or comparative language only when that exact
   relationship is explicitly supported by the context.
   Do not infer characteristics of one side of a comparison.

9. Identify every part of the question and answer each directly.
   If the question asks how something changed and the context
   explicitly provides a percentage or amount of change, include it.

10. Do not add unrelated background, alternative metrics,
    component breakdowns, examples, implications, caveats,
    or related facts unless they are required to answer the question.

11. Never state a fact and later claim that the retrieved context
    does not provide that same fact.

12. Cite factual claims using [SOURCE 1], [SOURCE 2], etc.
    Never cite a source that does not support the associated claim.

13. If any required part genuinely cannot be answered from the
    retrieved context, do not infer or reconstruct it.
    Say only:
    "I don't know based on the retrieved context."

EXACT ANSWER EXTRACTION:

- When the question asks for an equation, mathematical response,
  particular solution, limiting response, formula, or expression,
  and the retrieved context explicitly provides that expression,
  reproduce the expression exactly.

- Do not replace an explicit mathematical result with only a
  qualitative description.

- If the context says "the solution reduces to", "the particular
  solution is", "the response becomes", or equivalent wording,
  treat the following expression as the primary answer.

- Preserve every coefficient, variable, factor, sign, exponent,
  derivative order, subscript, and operator exactly.

{feedback}
"""
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])
rag_chain = gen_prompt | llm | StrOutputParser()

# Verifier (Anti-Hallucination Gate)
structured_verifier = llm.with_structured_output(
    CitationVerification,
    method="json_schema"
)
verify_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
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
"""
    )
])
verifier_chain = verify_prompt | structured_verifier

# ==========================================
# 3. DEFINE GRAPH NODES
# ==========================================
def retrieve_node(state):

    print(
        "--- 📂 NODE: HYBRID RETRIEVAL & RERANKING ---"
    )
    
    search_query = (
        state.get("search_query")
        or state["question"]
    )

    print(
        f"Search query: {search_query}"
    )
    with measure_stage("retrieval_ms"):
        use_wide = (
            state.get(
                "rewrite_count",
                0,
            ) > 0
        )

        if use_wide:
            print(
                "🔎 Using WIDE fallback retrieval."
            )

        docs = retrieve_with_neighbor_context(
            search_query,
            page_radius=1,
            max_docs=6 if use_wide else 4,
            use_wide=use_wide,
        )

    retrieved_documents = []

    for doc in docs:

        retrieved_documents.append({
            "content": doc.page_content,

            "retrieval_excerpt": doc.metadata.get(
                "retrieval_excerpt",
                doc.page_content,
            ),

            "source": doc.metadata.get(
                "source",
                "unknown"
            ),
            "page": doc.metadata.get(
                "page",
                -1
            ),
            "extraction_method":
                doc.metadata.get(
                    "extraction_method",
                    "unknown"
                ),
            "chunk_index": doc.metadata.get(
                "chunk_index",
                -1,
            ),
        })
    
    for i, doc in enumerate(
        retrieved_documents,
        start=1
    ):
        print(
            f"   [{i}] "
            f"{doc['source']} "
            f"| page {doc['page']} "
            f"| chunk {doc.get('chunk_index', -1)} "
            f"| {doc['extraction_method']}"
        )

    return {
        "documents": retrieved_documents,
        "question": state["question"]
    }

MAX_REWRITES = 1

def rewrite_node(state):

    print(
        "--- 🔄 NODE: REWRITING RETRIEVAL QUERY ---"
    )
    print(
        "rewrite_count =",
        state.get("rewrite_count", 0)
    )
    # Always rewrite from the ORIGINAL user question.
    with measure_stage("rewrite_ms"):
        result = question_rewriter.invoke(
            {
                "question":
                    state["question"]
            }
        )

    if isinstance(
        result,
        dict,
    ):
        better_query = result.get(
            "query",
            state["question"],
        )
    else:
        better_query = getattr(
            result,
            "query",
            state["question"],
        )

    better_query = (
        better_query.strip()
    )

    print(
        f"Original question: "
        f"{state['question']}"
    )

    print(
        f"Search rewrite: "
        f"{better_query}"
    )

    return {
        "search_query": better_query,
        "rewrite_count":
            state.get("rewrite_count", 0) + 1,
    }

def generate_node(state):
    print("--- 🤖 NODE: GENERATING ANSWER ---")
    context = format_context(state["documents"])
    
    feedback_msg = ""

    if state.get("verification_feedback"):

        print("⚠️ Applying feedback from verifier...")

        feedback_msg = f"""
    PREVIOUS DRAFT:
    {state["generation"]}

    VERIFIER FEEDBACK:
    {state["verification_feedback"]}

    Revise the previous draft rather than generating a completely
    new answer.

    Preserve all supported claims.

    For mathematical equations:
    - preserve the exact mathematical relationship in the source;
    - do not drop exponents, roots, signs, subscripts, or operators;
    - do not algebraically rearrange an equation unless necessary.
    """
        
    with measure_stage("generation_ms"):
        generation = rag_chain.invoke(
            {
                "context": context,
                "question": state["question"],
                "feedback": feedback_msg,
            }
        )
    return {"generation": generation}

def verify_node(state):

    print(
        "--- ⚖️ NODE: GROUNDING VERIFICATION ---"
    )

    generation = state["generation"]

    # --------------------------------------------------------
    # Existing abstention path
    # --------------------------------------------------------

    if generation_lacks_context(
        generation
    ):

        print(
            "✅ Insufficient-context answer detected. "
            "Returning clean fallback."
        )

        return {
            "generation": (
                "I'm sorry, the retrieved context does not contain "
                "enough verified information to safely answer "
                "this question."
            ),
            "verification_feedback": None,
        }

    # --------------------------------------------------------
    # Citation-local verification
    # --------------------------------------------------------

    cited_indices = (
        extract_cited_source_indices(
            generation
        )
    )

    if cited_indices:

        print(
            "🔎 Verifying only cited sources:",
            sorted(cited_indices),
        )

        context = format_context(
            state["documents"],
            selected_indices=cited_indices,
        )

    else:

        print(
            "⚠️ No citations detected; "
            "verifying against full context."
        )

        context = format_context(
            state["documents"]
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    with measure_stage(
        "verification_ms"
    ):

        result = verifier_chain.invoke(
            {
                "question":
                    state["question"],

                "context":
                    context,

                "generation":
                    generation,
            }
        )

    print(
        "Result:",
        (
            "✅ SUPPORTED"
            if result.is_supported
            else "❌ UNSUPPORTED"
        ),
    )

    print(
        "Core answer supported:",
        result.core_answer_supported,
    )

    # --------------------------------------------------------
    # Fully supported
    # --------------------------------------------------------

    if result.is_supported:

        return {
            "verification_feedback": None,
        }

    print(
        "Unsupported spans:",
        result.unsupported_claims,
    )

    print(
        "Reason:",
        result.reasoning,
    )

    # --------------------------------------------------------
    # Core answer itself is wrong -> FAIL CLOSED
    # --------------------------------------------------------

    if not result.core_answer_supported:

        print(
            "🛑 Core answer is unsupported. "
            "Returning safe fallback."
        )

        return {
            "generation": (
                "I'm sorry, the retrieved context does not contain "
                "enough verified information to safely answer "
                "this question."
            ),
            "verification_feedback": None,
        }

    # --------------------------------------------------------
    # Core answer is correct, optional material is bad.
    # Delete only exact unsupported text.
    # --------------------------------------------------------

    cleaned_generation = (
        redact_unsupported_claims(
            generation,
            result.unsupported_claims,
        )
    )

    if not cleaned_generation:

        print(
            "🛑 Safe redaction could not be performed. "
            "Returning fallback."
        )

        return {
            "generation": (
                "I'm sorry, the retrieved context does not contain "
                "enough verified information to safely answer "
                "this question."
            ),
            "verification_feedback": None,
        }

    print(
        "✂️ Unsupported optional material removed "
        "without regeneration."
    )

    return {
        "generation":
            cleaned_generation,

        "verification_feedback":
            None,
    }

def _doc_text(doc):
    """
    Return text from either our retrieved-document dict
    or another document-like object.
    """

    if isinstance(doc, dict):
        return doc.get(
            "content",
            doc.get(
                "page_content",
                "",
            ),
        )

    return getattr(
        doc,
        "page_content",
        str(doc),
    )

def _grader_doc_text(
    doc,
    max_chars=1200,
):
    """
    Use the child retrieval chunk for
    sufficiency grading.

    Fall back to parent content only when an
    excerpt is unavailable.
    """

    if isinstance(doc, dict):

        text = doc.get(
            "retrieval_excerpt"
        )

        if not text:
            text = _doc_text(doc)

    else:

        metadata = getattr(
            doc,
            "metadata",
            {},
        )

        text = metadata.get(
            "retrieval_excerpt"
        )

        if not text:
            text = _doc_text(doc)

    if len(text) <= max_chars:
        return text

    return text[:max_chars]

def decide_to_generate(state):
    print(
        "--- 🔀 DECISION: GRADING RETRIEVAL SUFFICIENCY ---"
    )

    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        print(
            "🛑 Rewrite already attempted."
        )
        return "generate"

    context = "\n\n".join(
        f"[DOCUMENT {index}]\n"
        f"{_grader_doc_text(doc)}"
        for index, doc in enumerate(
            state["documents"],
            start=1,
        )
    )

    with measure_stage("grading_ms"):
        grade = retrieval_grader.invoke(
            {
                "question": state["question"],
                "document": context,
            }
        )

    score = (
        grade.binary_score
        .strip()
        .lower()
    )

    if score == "yes":
        print(
            "✅ Retrieved context sufficient -> Generate"
        )
        return "generate"

    print(
        "❌ Retrieved context insufficient -> Rewrite"
    )

    return "rewrite"

def generation_lacks_context(
    generation: str,
) -> bool:

    text = (
        generation
        .lower()
        .strip()
    )

    markers = [
        "i don't know",
        "retrieved context does not contain enough",
        "context does not contain enough",
        "not enough information to answer",
        "insufficient information",
        "retrieved context does not provide enough",
    ]

    return any(
        marker in text
        for marker in markers
    )

def decide_after_generation(state):

    generation = state.get(
        "generation",
        "",
    )

    if generation_lacks_context(generation):

        if state.get("rewrite_count", 0) < MAX_REWRITES:
            print(
                "🔄 Generator detected insufficient context "
                "-> rewrite retrieval query"
            )
            return "rewrite"

        print(
            "🛑 Retrieval rewrite exhausted "
            "-> verify/fallback"
        )
        return "verify"

    return "verify"