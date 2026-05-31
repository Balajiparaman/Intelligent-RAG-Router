import os 
from typing import TypedDict, List, Dict, Any, Literal 
from pydantic import BaseModel, Field 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate 
from langgraph.graph import StateGraph, START, END 

from cost_aware_rag.database import query_database 

# 1. Define the shared state dictionary 
class AgentState(TypedDict):
    query: str 
    retreived_docs: List(Dict[str, Any])
    generated_response: str 
    citation_audit_report: Dict[str, Any] 
    revision_count: int 

# 2. Define the Pydantic schema for the Auditor's structured response
class AuditReport(BaseModel):
    is_valid: bool = Field(description="True if ALL facts in the response have valid, accurate footnotes, and the cited page text directly supports the claim. False if there are hallucinations, invalid page citations, or unsupported claims.")   
    invalid_citations: List[str] = Field(description="List of specific claims/citations that are incorrect or unsupported (eg 'Claim X cites Page 20 but Page 20 does not mention X').")
    feedback: str = Field(description="Detailed critique and instructions for the Researcher on how to correct any errors. Be constructive.")

# =====================================================================
# Node Functions
# =====================================================================

    
