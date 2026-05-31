from cost_aware_rag.router import route_query 

# We will test both casual and specific medical coding prompts 
test_queries = ["Hello there! Can you help me?",
"What is the coding guideline for a patient admitted with sepsis?",
"Tell me a joke about doctors.",
"When do I assign code U07.1 for coronavirus?",
"What is the capital of France?"]

if __name__ == "__main__":
    print("starting Intent Router Test...\n")
    for q in test_queries:
        print(f"Query: '{q}'")
        decision = route_query(q)
        print(f" -> INTENT: {decision.intent}")
        print(f" -> Confidence: {decision.confidence}")
        print(f" -> Reasoning: {decision.reasoning}")
        print("-" * 50)
        