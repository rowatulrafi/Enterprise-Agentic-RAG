from src.config import llm
from src.state import GradeDocuments

grader = llm.with_structured_output(
    GradeDocuments,
    method="json_schema"
)

result = grader.invoke(
    """
    Retrieved document:
    Dhaka is the capital city of Bangladesh.

    User question:
    What is the capital of Bangladesh?

    Determine whether this document contains enough information
    to answer the question.
    """
)

print(result)
print(type(result))
print(result.binary_score)