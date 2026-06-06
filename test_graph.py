from cost_aware_rag.graph import graph 

# Test with a complex coding rule query 
test_query = "What is the specific coding sequencing guideline for a pregnant patient who has COVID-19?"

if __name__ == "__main__":
    print(f"Running Stateful LangGraph Loop for query: '{test_query}'\n")
    
    initial_state = {
        "query": test_query,
        "revision_count": 0,
        "citation_audit_report": {"is_valid": True} # initial placeholder
    }
    
    final_state = graph.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL RESPONSE:")
    print("="*50)
    print(final_state["generated_response"])
    print("\nRevision Attempts:", final_state["revision_count"])
    print("Final Audit Report:", final_state["citation_audit_report"])
