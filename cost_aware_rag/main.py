import json 
import asyncio 
from fastapi import FastAPI 
from fastapi.responses import StreamingResponse 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel 
from langchain_google_genai import ChatGoogleGenerativeAI 

from cost_aware_rag.router import route_query 
from cost_aware_rag.graph import graph 

app = FastAPI(
    title="Intelligent RAG Router"
)

# Enable CORS (Cross-Origin Resource Sharing) so frontends can connect to our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str 

async def generate_chat_stream(message: str):
    """
    Asynchronous generator that yields Server-Sent Events (SSE).
    Steps:
    1. Classifies intent using the Router.
    2. If CHAT: Streams Gemini's response directly (bypassing DB).
    3. If RAG: Runs the LangGraph compliance citation loop.
    """
    loop = asyncio.get_event_loop() 

    # 1. Run the Intent Router in a background thread to prevent blocking
    yield f"data: {json.dumps({'event': 'status', 'message': 'Classifying intent...'})}\n\n"
    decision = await loop.run_in_executor(None, route_query, message)

    yield f"data: {json.dumps({
        'event': 'route', 
        'intent': decision.intent, 
        'reasoning': decision.reasoning
    })}\n\n"
    await asyncio.sleep(0.2)  # Tiny pause to let the client render the routing state   

    if decision.intent == "CHAT":
        yield f"data: {json.dumps({'event': 'status', 'message': 'Streaming direct chat response...'})}\n\n"
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

        # Stream the tokens directly from the cheap model
        async for chunk in llm.astream(message):
            yield f"data: {json.dumps({
                'event': 'token', 
                'text': chunk.content  # Corrected from context to content
            })}\n\n"

        # Shifted outside the async for loop
        yield f"data: {json.dumps({
            'event': 'done',
            'message': 'Stream complete'
        })}\n\n"

    else:
        # Run RAG Pipeline in a background thread
        yield f"data: {json.dumps({
            'event': 'status', 
            'message': 'Consulting guidelines database & running citation audit loop...'
        })}\n\n"

        initial_state = {
            "query": message,
            "revision_count": 0,
            "citation_audit_report": {"is_valid": True}  # placeholder
        }

        # Execute the LangGraph loop 
        final_state = await loop.run_in_executor(None, lambda: graph.invoke(initial_state))
        
        response_text = final_state.get("generated_response", "No response generated.")
        audit_report = final_state.get("citation_audit_report", {})
        revisions = final_state.get("revision_count", 0)

        # Stream the final audit-validated response text (Added this block back)
        yield f"data: {json.dumps({
            'event': 'token',
            'text': response_text
        })}\n\n"

        # Stream the final audit metadata
        yield f"data: {json.dumps({
            'event': 'audit',
            'is_valid': audit_report.get('is_valid', False),
            'revisions': revisions,
            'feedback': audit_report.get('feedback', '')
        })}\n\n"

        yield f"data: {json.dumps({'event': 'end'})}\n\n"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """POST endpoint that streams responses via Server-Sent Events."""
    return StreamingResponse(
        generate_chat_stream(request.message), 
        media_type="text/event-stream"
    )      
