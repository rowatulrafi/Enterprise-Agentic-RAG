from typing import List, Dict, Any, Optional
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
    documents: List[RetrievedDocument]
    generation: str
    verification_feedback: Optional[str]
    retries: int
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
    """Evaluates if the generated answer is strictly supported by context."""
    is_supported: bool = Field(description="True if claims are supported, False if hallucinations exist.")
    reasoning: str = Field(description="Explanation of unsupported claims, or confirmation of support.")

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


