import os
import json
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from src.config import llm, embeddings, cross_encoder, CHROMA_DIR, DATA_DIR
from src.state import GradeDocuments, RewrittenQuery, CitationVerification

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
    with open(bm25_data_path, "r") as f:
        text_data = json.load(f)
    bm25_retriever = BM25Retriever.from_texts(text_data)
    bm25_retriever.k = 3
    
    # Combine and Rerank
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5])
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
    hybrid_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
else:
    print("⚠️ BM25 Corpus not found. Falling back to pure Semantic Search until ingestion is run.")
    hybrid_retriever = semantic_retriever


# ==========================================
# 2. INITIALIZE LLM CHAINS
# ==========================================
# Grader (Agentic Document Relevance)
structured_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict grader. If the document provides enough context to answer the user's question, grade it 'yes'. Otherwise, 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
retrieval_grader = grade_prompt | structured_grader

# Rewriter (Agentic Query Optimization)
structured_rewriter = llm.with_structured_output(RewrittenQuery)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert search query generator. Transform the user's question into a highly optimized search string for a vector database. "
               "Extract core entities, convert spelled-out Greek letters to symbols (e.g. alpha to α, omega to ω), and drop conversational filler. "
               "Output ONLY the raw optimized query."),
    ("human", "Initial question: \n\n {question}")
])
question_rewriter = rewrite_prompt | structured_rewriter

# Generator (Drafting the Answer)
gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assistant. Use the provided context to answer. "
               "If a claim in the user's question is contradicted or not present in the context, explicitly state that it is false or missing, but answer the rest of the question if possible. "
               "If the context lacks the answer entirely, say 'I don't know'. {feedback}"),
    ("human", "Question: {question} \n\n Context: {context}")
])
rag_chain = gen_prompt | llm | StrOutputParser()

# Verifier (Anti-Hallucination Gate)
structured_verifier = llm.with_structured_output(CitationVerification)
verify_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict fact-checker. Verify that EVERY claim in the answer is explicitly stated in the context. If it hallucinates or uses outside knowledge, fail it."),
    ("human", "Context: {context} \n\n Answer to Verify: {generation}")
])
verifier_chain = verify_prompt | structured_verifier

# ==========================================
# 3. DEFINE GRAPH NODES
# ==========================================
def retrieve_node(state):
    print("--- 📂 NODE: HYBRID RETRIEVAL & RERANKING ---")
    docs = hybrid_retriever.invoke(state["question"])
    return {"documents": [doc.page_content for doc in docs], "question": state["question"]}

def rewrite_node(state):
    print("--- 🔄 NODE: REWRITING QUERY ---")
    result = question_rewriter.invoke({"question": state["question"]})
    
    better_question = getattr(result, "query", str(result)) if not isinstance(result, dict) else result.get("query", str(result))
    print(f"Rewritten: {better_question}")
    
    return {"question": better_question, "retries": state.get("retries", 0) + 1}

def generate_node(state):
    print("--- 🤖 NODE: GENERATING ANSWER ---")
    context = "\n\n".join(state["documents"])
    
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

    context = "\n\n".join(state["documents"])
    result = verifier_chain.invoke({"context": context, "generation": state["generation"]})
    
    print(f"Result: {'✅ SUPPORTED' if result.is_supported else '❌ HALLUCINATION'}")
    print(f"Reasoning: {result.reasoning}")
    
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

    for doc_text in state["documents"]:
        grade = retrieval_grader.invoke({"question": state["question"], "document": doc_text})
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