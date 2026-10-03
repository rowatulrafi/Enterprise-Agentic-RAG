from langgraph.graph import END, StateGraph
from src.state import MasterRAGState
from src.nodes import (
    retrieve_node, rewrite_node, generate_node, verify_node,
    decide_to_generate, decide_after_generation,
)

# 1. Initialize the State Machine
workflow = StateGraph(MasterRAGState)

# 2. Add the computational nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("verify", verify_node)

# 3. Map the Edges (The Logic Wires)
workflow.set_entry_point("retrieve")

# After retrieval, grade the documents to decide where to go
workflow.add_conditional_edges(
    "retrieve",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

# If rewritten, go back to retrieve
workflow.add_edge("rewrite", "retrieve")

# After generation, always verify
workflow.add_conditional_edges(
    "generate",
    decide_after_generation,
    {
        "rewrite": "rewrite",
        "verify": "verify",
    },
)

# After verification, decide if we can end or need to re-generate
workflow.add_edge(
    "verify",
    END
)

# 4. Compile the application
app = workflow.compile()