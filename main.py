from src.graph import app

def main():
    print("\n" + "="*60)
    print("LOCAL ENTERPRISE AGENTIC RAG")
    print("="*60 + "\n")
    print("LLM       : Qwen3 8B via LM Studio" + "\n")
    print("Retrieval : Dense + BM25 + Cross-Encoder Reranking" + "\n")
    print("Safety    : Query Rewrite + Grounding Verification" + "\n")
    print("="*60 + "\n")

    while True:
        user_input = input("\n🧑‍💻 You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down pipeline...")
            break

        inputs = {"question": user_input, "retries": 0, "verification_feedback": None}
        final_answer = ""
        
        # Stream the LangGraph execution
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"[{key} completed]")
                
                # Capture the final generation
                if "generation" in value:
                    final_answer = value["generation"]
                    
            print("-" * 50)
            
        print("\n💡 FINAL VERIFIED ANSWER:\n", final_answer)

if __name__ == "__main__":
    main()