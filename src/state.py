from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ==========================================
# 1. DATA INGESTION & VALIDATION SCHEMAS
# ==========================================
class RetrievedDocument(TypedDict):
    content: str
    source: str
    page: int
    extraction_method: str

class DocumentValidation(BaseModel):
    """Evaluates extracted OCR text for coherence and structure."""
    confidence_score: int = Field(description="Score from 0 to 100 indicating readability.")
    is_garbled: bool = Field(description="True if the text is random characters or broken OCR.")
    reasoning: str = Field(description="Brief explanation for the score without raw control characters.")

# ==========================================
# 2. MASTER LANGGRAPH RAG STATE
# ==========================================
class MasterRAGState(TypedDict):
    question: str
    search_query: str
    documents: List[RetrievedDocument]
    generation: str
    verification_feedback: Optional[str]

    rewrite_count: int
    verification_retries: int
    
# ==========================================
# 3. STRUCTURED LLM OUTPUTS FOR GRAPH
# ==========================================
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class RewrittenQuery(BaseModel):
    """The rewritten query for vector search."""
    query: str = Field(description="The optimized search query string ONLY.")

class CitationVerification(BaseModel):
    """Compact grounding verification result."""

    is_supported: bool = Field(
        description=(
            "True only if every factual claim in the answer "
            "is explicitly supported by the retrieved context."
        )
    )

    unsupported_claims: List[str] = Field(
        default_factory=list,
        description=(
            "Only factual claims that are unsupported or "
            "incorrectly grounded. Empty when fully supported."
        )
    )

    reasoning: str = Field(
        description=(
            "One short sentence explaining the decision."
        )
    )


