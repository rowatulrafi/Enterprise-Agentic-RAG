import os
import json

from src.telemetry import measure_stage
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

GENERIC RULES:
- Answer only what is necessary to answer the question.
- Do not add unrelated caveats, alternative metrics, or extra facts unless they are necessary for disambiguation.

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
You are a strict grounding verifier.

Check whether EVERY factual claim in the answer is explicitly
supported by the retrieved context.

Mark the answer UNSUPPORTED if any claim:

- uses the wrong number, date, unit, entity, or qualifier;
- attaches a value to the wrong methodology or metric;
- presents an example-specific value as a general rule;
- makes a causal or comparative claim not stated in context;
- changes the meaning of a mathematical equation;
- changes an exponent, root, sign, subscript, operator, or equality;
- cites a source that does not support the claim;
- introduces outside knowledge, including in examples or parentheses.

If the answer says the context lacks information but then supplies
that missing information anyway, mark it unsupported.

Return:
- is_supported
- unsupported_claims: ONLY unsupported claims
- reasoning: one short sentence

If all claims are supported, unsupported_claims must be empty.
"""
    ),
    (
        "human",
        """
Retrieved context:
{context}

Answer:
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

    print("--- ⚖️ NODE: GROUNDING VERIFICATION ---")

    generation = state["generation"]

    # Do not waste another LLM call verifying an abstention.
    if generation_lacks_context(generation):

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

    context = format_context(
        state["documents"]
    )

    with measure_stage("verification_ms"):

        result = verifier_chain.invoke(
            {
                "context": context,
                "generation": generation,
            }
        )

    print(
        f"Result: "
        f"{'✅ SUPPORTED' if result.is_supported else '❌ UNSUPPORTED'}"
    )

    if result.is_supported:

        return {
            "verification_feedback": None,
        }

    print(
        "Unsupported claims:",
        result.unsupported_claims,
    )

    print(
        "Reason:",
        result.reasoning,
    )

    print(
        "🛑 Returning safe fallback."
    )

    return {
        "generation": (
            "I'm sorry, the retrieved context does not contain "
            "enough verified information to safely answer "
            "this question."
        ),
        "verification_feedback": None,
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