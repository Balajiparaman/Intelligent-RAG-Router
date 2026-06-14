import os 
from typing import TypedDict, List, Dict, Any, Literal 
from pydantic import BaseModel, Field 
from langchain_ollama import ChatOllama 
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
    docs = query_database(state["query"], n_results=2)
    return {"retrieved_docs": docs}

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Researcher synthesizes the coding guidance based on context and feedback."""
    print("\n[Node: Researcher] Writing coding response...")

    # Initialize the LLM
    llm = ChatOllama(model="clinical-coder", temperature=0.2)

    # Format the retrieved pages for context - loop ONLY formats text
    context = ""
    for doc in state["retrieved_docs"]:
        context += f"\n--- START OF PAGE {doc['page_num']} ---\n{doc['text']}\n--- END OF PAGE {doc['page_num']} ---\n"

    # Now define prompt outside the loop
    system_prompt = (
        "You are an expert ICD-10-CM Medical Coding Researcher.\n"
        "Your task is to answer the user's coding query using ONLY the guidelines provided below. Do not assume or extrapolate.\n\n"
        "Guideline Search Instructions:\n"
        "1. Carefully scan the guideline pages below to find the specific section matching the user's request (e.g. if looking for 'COVID-19 in pregnancy', look for 'COVID-19 infection in pregnancy').\n"
        "2. Only extract rules and codes from that specific matching section. Do NOT mention or include rules or codes from unrelated sections (like abuse complicating pregnancy, MRSA, or other conditions) even if they are on the same page. Focus ONLY on the queried topic.\n"
        "3. Quote the codes (e.g. O98.5-, U07.1) exactly as they appear in the matching section.\n\n"
        "Citation Rule:\n"
        "For every rule or code you state, append its page number in brackets, for example: [Page X]. Only use page numbers listed below.\n\n"
        "If the guidelines do not contain the answer, say 'I don't know'.\n\n"
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
    llm = ChatOllama(model="clinical-coder", temperature=0.0)
    structured_llm = llm.with_structured_output(AuditResult)

    # Format context pages for the Auditor - loop ONLY formats text
    context = ""
    for doc in state["retrieved_docs"]:
        context += f"\n[Page {doc['page_num']} Raw Text]:\n{doc['text']}\n"

    # Define system prompt outside the loop
    system_prompt = (
        "You are a strict Medical Coding Citation Compliance Auditor.\n"
        "Compare the Researcher's response with the provided raw page guidelines to verify citation alignment.\n\n"
        "Verification Checklist:\n"
        "1. Check all bracketed page citations (e.g., [Page X]). Are those page numbers present in the raw text?\n"
        "2. Read the raw text of the cited page. Does it support the researcher's claim?\n"
        "3. If the researcher cites a code or section name (e.g., 'Section II.H') that is NOT in the raw text, flag it as invalid.\n\n"
        "Provide constructive feedback only for the invalid parts, and set is_valid to False if there are any mismatches.\n\n"
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
