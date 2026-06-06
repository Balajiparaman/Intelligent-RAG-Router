import os 
from typing import TypedDict, List, Dict, Any, Literal 
from pydantic import BaseModel, Field 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate 
from langgraph.graph import StateGraph, START, END 
from dotenv import load_dotenv  # <-- Add this import

from cost_aware_rag.database import query_database 

load_dotenv()

# 1. Define the shared state dictionary 
class AgentState(TypedDict):
    query: str 
    retrieved_docs: List[Dict[str, Any]]  # Corrected brackets and spelling
    generated_response: str 
    citation_audit_report: Dict[str, Any] 
    revision_count: int 

# 2. Define the Pydantic schema for the Auditor's structured response
class AuditResult(BaseModel):  # Named AuditResult to match usages
    is_valid: bool = Field(description="True if ALL facts in the response have valid, accurate footnotes, and the cited page text directly supports the claim. False if there are hallucinations, invalid page citations, or unsupported claims.")   
    invalid_citations: List[str] = Field(description="List of specific claims/citations that are incorrect or unsupported.")
    feedback: str = Field(description="Detailed critique and instructions for the Researcher on how to correct any errors. Be constructive.")

# =====================================================================
# Node Functions
# =====================================================================

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant guideline pages from ChromaDB."""
    print("\n[Node: Retrieve] Fetching relevant ICD-10 pages from ChromaDB...")
    docs = query_database(state["query"], n_results=4)
    return {"retrieved_docs": docs}

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Researcher synthesizes the coding guidance based on context and feedback."""
    print("\n[Node: Researcher] Writing coding response...")

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # Format the retrieved pages for context - loop ONLY formats text
    context = ""
    for doc in state["retrieved_docs"]:
        context += f"\n--- START OF PAGE {doc['page_num']} ---\n{doc['text']}\n--- END OF PAGE {doc['page_num']} ---\n"

    # Now define prompt outside the loop
    system_prompt = (
        "You are an expert ICD-10-CM Medical Coding Researcher.\n"
        "Your task is to answer the user's coding query using ONLY the provided guideline pages.\n\n"
        "Strict Citation Rules:\n"
        "1. For EVERY fact, code, or instruction you state, you MUST add a inline citation referencing the page number, e.g. [Page X].\n"
        "2. Do NOT mention page numbers that are not present in the context.\n"
        "3. If the context does not contain enough information to answer, state clearly: 'I don't know' or 'Guidelines for this were not found in the retrieved sections.'\n\n"
        f"--- RETRIEVED GUIDELINE PAGES ---\n{context}"
    )

    messages = [("system", system_prompt)]

    # If this is a revision loop, inject the auditor's feedback
    audit = state.get("citation_audit_report")
    if audit and not audit.get("is_valid"):
        feedback_msg = (  
            f"Your previous attempt was rejected by the Auditor due to citation errors:\n"
            f"Critique: {audit['feedback']}\n"
            f"Invalid items: {audit['invalid_citations']}\n\n"
            "Please revise your answer: correct the citations, remove unsupported claims, "
            "and ensure everything aligns strictly with the raw text."
        )
        messages.append(("human", feedback_msg))
    else:
        messages.append(("human", state["query"]))  # Corrected append parameters

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm
    response = chain.invoke({})

    return {"generated_response": response.content}  # Corrected property to content

def auditor_node(state: AgentState) -> Dict[str, Any]:
    """Auditor verifies the Researcher's response for citation accuracy."""
    print("\n[Node: Auditor] Auditing citations and factual alignment...")

    # Initialize the LLM with structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    structured_llm = llm.with_structured_output(AuditResult)

    # Format context pages for the Auditor - loop ONLY formats text
    context = ""
    for doc in state["retrieved_docs"]:
        context += f"\n[Page {doc['page_num']} Raw Text]:\n{doc['text']}\n"

    # Define system prompt outside the loop
    system_prompt = (
        "You are a strict Medical Coding Citation Compliance Auditor.\n"
        "Your task is to verify that the Researcher's generated response is 100% accurate and "
        "has valid citations based ONLY on the provided raw pages.\n\n"
        "Verification Checklist:\n"
        "1. Check every page cited in brackets (e.g. [Page X]) in the response. Is it present in the raw text?\n"
        "2. For each cited fact, read the raw text of that specific page. Does it actually support the claim? "
        "If the Researcher claims a code is required but the page text doesn't say so, mark it as invalid.\n"
        "3. If the Researcher cited a page that wasn't provided, mark it as invalid.\n\n"
        f"--- RAW GUIDELINE PAGES ---\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Generated Response to Audit:\n{state['generated_response']}")
    ])

    chain = prompt | structured_llm 
    audit_result: AuditResult = chain.invoke({})
    
    # Increment the revision count 
    current_revisions = state.get("revision_count", 0)
    
    return {
        "citation_audit_report": audit_result.model_dump(),
        "revision_count": current_revisions + 1
    }

# =====================================================================
# StateGraph Definition
# =====================================================================

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("auditor", auditor_node)

# Connect edges 
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "researcher")
workflow.add_edge("researcher", "auditor")

# Conditional routing edge from Auditor 
def route_after_audit(state: AgentState) -> Literal["researcher", "__end__"]:
    audit = state["citation_audit_report"]
    revisions = state["revision_count"]  # Renamed variable to match checks

    if audit["is_valid"]:
        print("\n[Auditor Status]: PASSED. Response verified successfully.")
        return "__end__"

    if revisions >= 3:
        print("\n[Auditor Status]: FAILED but max revision attempts (3) reached. Exiting to avoid infinite loop.")
        return "__end__"

    print(f"\n[Auditor Status]: REJECTED. Routing back to Researcher (Attempt {revisions}/3).")
    return "researcher"

# Corrected indentation (shifted outside of route_after_audit)
workflow.add_conditional_edges(
    "auditor",
    route_after_audit,
    {
        "researcher": "researcher",
        "__end__": END
    }
)

# Compile the final graph 
graph = workflow.compile()
