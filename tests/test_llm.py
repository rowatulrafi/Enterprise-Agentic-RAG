from src.config import llm

response = llm.invoke(
    "Reply with exactly: QWEN CONNECTED"
)

print(response.content)