import os
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from src.config import llm, embeddings, cross_encoder, CHROMA_DIR, DATA_DIR
from src.state import GradeDocuments, RewrittenQuery, CitationVerification
from src.retrieval import hybrid_rerank_retriever
# ==========================================
# 0. Helper for formatting context
# ==========================================
def format_context(documents):

    formatted_sections = []

    for index, doc in enumerate(
        documents,
        start=1
    ):

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

You receive a user's question and the complete retrieved
context that would be supplied to the answer generator.

Return YES only if the retrieved context, taken together,
contains enough explicit evidence to answer the exact
question faithfully.

Return NO when:
- passages are merely about the same topic;
- requested numbers, qualifiers or relationships are absent;
- related variables are present but the requested relation is absent;
- an equation is requested but that equation is absent;
- answering requires outside knowledge or inference;
- an example is present but the requested general rule is absent.

For mathematical text, imperfect extraction is acceptable.
Equations and symbols count as evidence when they actually
support the requested relationship.

Do not answer the question.
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

GROUNDING RULES:
1. Every factual claim must be explicitly supported by the context.
2. Preserve exact entity, year, date, quantity, comparison, and qualifier relationships.
3. Never transfer a statement about one year, entity, or metric to another.
4. Never infer causation unless the context explicitly states the causal relationship.
5. Do not combine separate facts in a way that changes their meaning.
6. If information is missing, explicitly say that the context does not provide it.
7. If the context cannot answer the question, say "I don't know".
8. Cite factual claims using the source labels provided in the context.
9. Use the format [SOURCE 1], [SOURCE 2], etc.
10. Never cite a source that does not support the associated statement.
13. Preserve qualifiers exactly as they appear in the source.
14. Never attach a methodology, definition, date, unit, status,
    or qualifier to a number unless the source explicitly associates
    that qualifier with that number.
15. Distinguish general rules from document-specific examples.
    Example parameter values must never be presented as universal rules.
16. When a source gives multiple versions of the same metric
    (for example gross reserves vs BPM6-compliant reserves),
    keep the metric names and values strictly paired.
17. If the retrieved context lacks a fact, do not provide that fact
even as an example, clarification, parenthetical, common formula,
or outside-knowledge note.

GROUNDING RULES:

1. QUALIFIER BINDING
A qualifier belongs only to the value/entity it explicitly
modifies. Never transfer terms such as gross, net, BPM6,
provisional, revised, annual, monthly, FY25 or FY26 to a
neighboring value.

2. GENERAL RULE VS EXAMPLE
Never present example-specific values as universal rules.
If an example is useful, explicitly label it as an example.
For mathematical expressions, preserve the exact equation structure provided by the source. Do not drop or alter exponents, roots, signs, subscripts, or operators.

3. CAUSALITY
Use causal language only when the context explicitly states
a causal relationship between the SAME variables.

4. SOURCE TERMINOLOGY
Prefer terminology actually used in the retrieved context.
Do not invent related technical or conceptual labels.

5. SCOPE
Do not combine facts from different entities, years,
methodologies or measurement definitions as if they belong
to one statement.

6. MISSING EVIDENCE
If retrieved context is insufficient, say that the RETRIEVED
CONTEXT does not contain enough information. Do not claim
that the entire source document lacks the information.

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
You are a strict claim-level grounding verifier.

Break the generated answer into individual factual claims and verify
each one against the provided context.

A claim is SUPPORTED only when the context explicitly entails the
complete claim.

Pay special attention to:

- years and dates
- numbers and units
- entities
- comparisons
- increases/decreases
- superlatives such as "highest" or "three-year high"
- causal statements such as "because of" or "driven by"
- relationships between separate facts

For every claim verify:
- entity binding
- year/date binding
- number/value binding
- methodology/qualifier binding
- general rule versus example
- causal relationship
- scope
- citation support

Adjacent numbers are NOT interchangeable.

A claim fails if its number is correct but attached to the
wrong qualifier, methodology, year or entity.

A claim fails if example-specific values are presented as a
general rule.

A claim fails if causal language is stronger than the source.

For mathematical claims, verify every exponent, root, sign, subscript, operator, and equality relation. A mathematically different expression MUST be marked unsupported.

MISSING-EVIDENCE CONSISTENCY:

Any factual content introduced after phrases such as
"for example", "e.g.", "typically", "generally", "commonly",
or "such as" must also be explicitly supported by the context.

If an answer says the context lacks some information but then
supplies that missing information from model knowledge, the
answer MUST be marked unsupported.

CRITICAL RULE:
Never transfer a description, comparison, cause, number, or qualifier
from one year/entity to another.

Example:
If the context says:
"2025 reserves reached a three-year high"
and
"2024 reserves were $26 billion"

then the claim:
"2024 reserves represented a three-year high"

MUST be marked unsupported.

QUALIFIER BINDING:

A claim is unsupported if the main number is correct but any
qualifier attached to it is incorrect.

Verify independently:
- metric definition
- methodology
- date/year
- unit
- entity
- comparison
- cause
- scope

Example:

Context:
Gross reserves in 2024 = $26.21B.
BPM6 reserves in 2024 = $21.39B.

Claim:
"BPM6 reserves were $26.21B in 2024."

This MUST be marked unsupported.

GENERAL RULE VS EXAMPLE:

If the source gives an example using specific parameter values,
those values must remain explicitly scoped to that example.

Do not treat example-specific values as universal conditions.

Set is_supported=true ONLY if every factual claim is supported.
"""
    ),
    (
        "human",
        """
Context:
{context}

Answer to verify:
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

    docs = hybrid_rerank_retriever.invoke(
        search_query
    )

    retrieved_documents = []

    for doc in docs:

        retrieved_documents.append({
            "content": doc.page_content,
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
        })
    
    for i, doc in enumerate(
        retrieved_documents,
        start=1
    ):
        print(
            f"   [{i}] "
            f"{doc['source']} "
            f"| page {doc['page']} "
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
        
    generation = rag_chain.invoke({"context": context, "question": state["question"], "feedback": feedback_msg})
    return {"generation": generation}

def verify_node(state):
    print("--- ⚖️ NODE: CITATION VERIFICATION ---")
    if "I don't know" in state["generation"]:
        print("✅ Safe fallback detected. Skipping verification.")
        return {
            "verification_feedback": None,
        }

    context = format_context(state["documents"])
    result = verifier_chain.invoke({"context": context, "generation": state["generation"]})
    
    print(
    f"Result: {'✅ SUPPORTED' if result.is_supported else '❌ UNSUPPORTED'}"
)

    for claim in result.claims:
        status = "✅" if claim.is_supported else "❌"
        print(f"{status} Claim: {claim.claim}")
        print(f"   Evidence: {claim.evidence}")

    print(f"Overall reasoning: {result.reasoning}")
    
    if result.is_supported:
        return {"verification_feedback": None}
    else:
        # THE SAFETY WIPE: If we fail verification at the max retry limit, wipe the answer!
        if state.get("verification_retries", 0) >= 1:
            print("🛑 Safety Valve: Wiping unsupported draft.")
            safe_fallback = (
                "I'm sorry, the retrieved context does not contain "
                "enough verified information to safely answer this question."
            )
            return {
                "generation": safe_fallback,
                "verification_feedback": None,
            }

        return {
            "verification_feedback": result.reasoning,
            "verification_retries":
                state.get("verification_retries", 0) + 1,
        }
# ==========================================
# 4. DEFINE ROUTING LOGIC
# ==========================================
def _doc_text(doc):
    if isinstance(doc, dict):
        return doc.get(
            "content",
            doc.get("page_content", "")
        )

    return str(doc)


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
        f"[DOCUMENT {index}]\n{_doc_text(doc)}"
        for index, doc in enumerate(
            state["documents"],
            start=1,
        )
    )

    grade = retrieval_grader.invoke(
        {
            "question":
                state["question"],

            "document":
                context,
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
    ]

    return any(
        marker in text
        for marker in markers
    )

def decide_verification(state):

    generation = state.get(
        "generation",
        "",
    )
    print(
        "rewrite_count =",
        state.get("rewrite_count", 0)
    )
    # The relevance grader can make a false-positive.
    # Give retrieval one second chance.
    if generation_lacks_context(
        generation
    ):

        if (
            state.get(
                "rewrite_count",
                0,
            )
            < MAX_REWRITES
        ):
            print(
                "🔄 Generator found insufficient "
                "context -> Rewrite retrieval query"
            )

            return "rewrite"

        print(
            "🛑 Retrieval rewrite already exhausted."
        )

        return "end"

    if (
        state.get(
            "verification_feedback"
        )
        is None
    ):
        return "end"

    print(
        "🔄 Re-routing to generator "
        "for grounded correction"
    )

    return "generate"