from langgraph.graph import END, StateGraph

from src.nodes import (
    decide_after_generation,
    decide_to_generate,
    generate_node,
    retrieve_node,
    rewrite_node,
    verify_node,
)
from src.state import MasterRAGState


workflow = StateGraph(MasterRAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("verify", verify_node)

workflow.set_entry_point("retrieve")

# Retrieval sufficiency determines whether to answer
# immediately or perform one query rewrite.
workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

workflow.add_edge(
    "rewrite",
    "retrieve",
)

# A generator-level insufficient-context decision can trigger
# the rewrite path; otherwise the answer is verified.
workflow.add_conditional_edges(
    "generate",
    decide_after_generation,
    {
        "rewrite": "rewrite",
        "verify": "verify",
    },
)

# Verification is the terminal safety gate.
workflow.add_edge(
    "verify",
    END,
)

app = workflow.compile()