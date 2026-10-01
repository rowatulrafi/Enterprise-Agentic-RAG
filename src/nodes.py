import os
import json
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document

from src.config import llm, embeddings, cross_encoder, CHROMA_DIR, DATA_DIR
from src.state import GradeDocuments, RewrittenQuery, CitationVerification

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
# Load Semantic Database
chroma_store = Chroma(
    persist_directory=CHROMA_DIR, 
    embedding_function=embeddings, 
    collection_name="ultimate_docs"
)
semantic_retriever = chroma_store.as_retriever(search_kwargs={"k": 3})

# Load Keyword Database (BM25)
bm25_data_path = os.path.join(DATA_DIR, "bm25_corpus.json")
if os.path.exists(bm25_data_path):

    with open(
        bm25_data_path,
        "r",
        encoding="utf-8"
    ) as f:
        bm25_data = json.load(f)

    bm25_documents = [
        Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        for item in bm25_data
    ]

    bm25_retriever = BM25Retriever.from_documents(
        bm25_documents
    )

    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            semantic_retriever
        ],
        weights=[0.5, 0.5]
    )

    compressor = CrossEncoderReranker(
        model=cross_encoder,
        top_n=3
    )

    hybrid_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

else:

    print(
        "⚠️ BM25 corpus not found. "
        "Falling back to semantic retrieval."
    )

    hybrid_retriever = semantic_retriever

# ==========================================
# 2. INITIALIZE LLM CHAINS
# ==========================================
# Grader (Agentic Document Relevance)
structured_grader = llm.with_structured_output(
    GradeDocuments,
    method="json_schema"
)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict grader. If the document provides enough context to answer the user's question, grade it 'yes'. Otherwise, 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
retrieval_grader = grade_prompt | structured_grader

# Rewriter (Agentic Query Optimization)
structured_rewriter = llm.with_structured_output(
    RewrittenQuery,
    method="json_schema"
)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert search query generator. Transform the user's question into a highly optimized search string for a vector database. "
               "Extract core entities, convert spelled-out Greek letters to symbols (e.g. alpha to α, omega to ω), and drop conversational filler. "
               "Output ONLY the raw optimized query."),
    ("human", "Initial question: \n\n {question}")
])
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

    docs = hybrid_retriever.invoke(
        state["question"]
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

def rewrite_node(state):
    print("--- 🔄 NODE: REWRITING QUERY ---")
    result = question_rewriter.invoke({"question": state["question"]})
    
    better_question = getattr(result, "query", str(result)) if not isinstance(result, dict) else result.get("query", str(result))
    print(f"Rewritten: {better_question}")
    
    return {"question": better_question, "retries": state.get("retries", 0) + 1}

def generate_node(state):
    print("--- 🤖 NODE: GENERATING ANSWER ---")
    context = format_context(state["documents"])
    
    feedback_msg = ""
    if state.get("verification_feedback"):
        print("⚠️ Applying feedback from verifier...")
        feedback_msg = f"\n\nCRITICAL FEEDBACK ON PREVIOUS DRAFT: {state['verification_feedback']}. Fix the hallucinations."
        
    generation = rag_chain.invoke({"context": context, "question": state["question"], "feedback": feedback_msg})
    return {"generation": generation}

def verify_node(state):
    print("--- ⚖️ NODE: CITATION VERIFICATION ---")
    if "I don't know" in state["generation"]:
        print("✅ Safe fallback detected. Skipping verification.")
        return {"verification_feedback": None}

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
        if state.get("retries", 0) >= 2:
            print("🛑 Safety Valve: Wiping hallucinated draft to protect user.")
            safe_fallback = "I'm sorry, the retrieved context does not contain enough verified information to safely answer this question."
            return {"generation": safe_fallback, "verification_feedback": None}
            
        return {"verification_feedback": result.reasoning, "retries": state.get("retries", 0) + 1}
# ==========================================
# 4. DEFINE ROUTING LOGIC
# ==========================================
def decide_to_generate(state):
    print("--- 🔀 DECISION: GRADING DOCUMENTS ---")
    if state.get("retries", 0) >= 2:
        print("🛑 Max retries reached. Forcing generation.")
        return "generate"

    for doc in state["documents"]:
        grade = retrieval_grader.invoke({"question": state["question"], "document": doc["content"]})
        if grade.binary_score.lower() == "yes":
            print("✅ Relevant documents found -> Proceed to Generate")
            return "generate"
            
    print("❌ Documents irrelevant -> Route to Rewrite")
    return "rewrite"

def decide_verification(state):
    if state.get("verification_feedback") is None:
        return "end"
    if state.get("retries", 0) >= 2:
        print("🛑 Max corrections reached. Ending loop.")
        return "end"
    print("🔄 Re-routing to Generator for correction")
    return "generate"