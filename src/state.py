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

class ClaimCheck(BaseModel):
    """Verification result for one factual claim."""
    
    claim: str = Field(
        description="The factual claim extracted from the generated answer."
    )
    
    is_supported: bool = Field(
        description="True only if the exact claim is explicitly supported by the context."
    )
    
    evidence: str = Field(
        description="The exact supporting evidence from context, or an explanation if unsupported."
    )
    source_ids: List[int] = Field(
        default_factory=list,
        description="Sources that directly support the complete claim."
    )

    error_type: Optional[str] = Field(
        default=None,
        description=(
            "If unsupported: number, date, entity, methodology, "
            "qualifier, causality, scope, citation, or other."
        )
    )

class CitationVerification(BaseModel):
    """Claim-level grounding verification."""
    
    is_supported: bool = Field(
        description="True only if every factual claim in the answer is supported."
    )
    
    claims: List[ClaimCheck] = Field(
        description="Verification result for each factual claim."
    )
    
    reasoning: str = Field(
        description="Overall explanation of the verification decision."
    )


