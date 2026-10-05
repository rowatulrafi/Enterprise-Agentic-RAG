from src.graph import app


def main():
    print("\n" + "=" * 60)
    print("LOCAL ENTERPRISE AGENTIC RAG")
    print("=" * 60)
    print("LLM       : Qwen3 8B via LM Studio")
    print("Retrieval : Dense + BM25 + Cross-Encoder Reranking")
    print("Safety    : Query Rewrite + Grounding Verification")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("Shutting down pipeline...")
            break

        if not user_input:
            continue

        inputs = {
            "question": user_input,
            "rewrite_count": 0,
        }

        final_answer = ""

        for output in app.stream(inputs):
            for node_name, value in output.items():
                print(f"[{node_name} completed]")

                if "generation" in value:
                    final_answer = value["generation"]

            print("-" * 50)

        print("\nFINAL VERIFIED ANSWER:\n")
        print(final_answer)


if __name__ == "__main__":
    main()