import requests
import json
import sys

url = "http://127.0.0.1:8000/chat"

# We will test one chat query and one RAG query
queries = [
    "Hello! Can you help me learn about medical coding?",
    "What is the coding sequencing guideline for a pregnant patient with COVID-19?"
]

def run_streaming_test(query: str):
    print(f"\nSending Query: '{query}'")
    print("=" * 60)
    
    # Send POST request with stream=True
    response = requests.post(url, json={"message": query}, stream=True)
    
    # Read the stream line-by-line
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                # Strip the "data: " prefix and load JSON
                event_data = json.loads(decoded_line[6:])
                event_type = event_data.get("event")
                
                if event_type == "status":
                    print(f"\n[Status]: {event_data.get('message')}")
                elif event_type == "route":
                    print(f"[Router]: Intent = {event_data.get('intent')} | Reasoning = {event_data.get('reasoning')}\n")
                elif event_type == "token":
                    # Print tokens token-by-token (without newlines)
                    sys.stdout.write(event_data.get("text", ""))
                    sys.stdout.flush()
                elif event_type == "audit":
                    print(f"\n\n[Auditor]: Valid = {event_data.get('is_valid')} | Revisions = {event_data.get('revisions')}")
                    if event_data.get("feedback"):
                        print(f"Feedback: {event_data.get('feedback')}")
                elif event_type == "done":
                    print("\n[Chat Stream Finished]")
                elif event_type == "end":
                    print("\n[RAG Stream Finished]")

if __name__ == "__main__":
    print("Starting Streaming API Client Tests...")
    
    # Test 1: Direct Chat Path
    run_streaming_test(queries[0])
    
    print("\n" + "#" * 60 + "\n")
    
    # Test 2: RAG / Citation Loop Path
    run_streaming_test(queries[1])
