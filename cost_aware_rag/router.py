from importlib import readers
import os 
from pydantic import BaseModel, Field 
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv 

# Load the API key from the .env file
load_dotenv()

 #Define the structured output format using Pydantic
class RoutingDecision(BaseModel):
    intent: str = Field(description="The classification intent. Must be either 'CHAT' or 'RAG'.")
    
    confidence: float = Field(description="Confidence score of the routing decision, between 0.0 and 1.0.")

    reasoning: str = Field(description="A brief, one-sentence explanation of why this route was chosen.")

def route_query(query: str) -> RoutingDecision:
    """
    Classifies user query and decides whether to use RAG or Chat
    using Ollama, cost aware

    Args:
        query: User query to classify

    Returns:
        RoutingDecision object with intent, confidence, and reasoning
    """

    # 1. Initialize our LLM (temperature=0.0 makes the routing deterministic)
    llm = ChatOllama(model="clinical-coder", temperature=0.0)

    # 2. Bind the pydantic schema to force the model to respond in JSON matching our schema
    structured_llm = llm.with_structured_output(RoutingDecision)

    # 3. Create the classification prompt
    system_prompt = (
        " You are an intent routing gatekeeper for an ICD-10-CM medical coding assistant.\n"
        "Your task is to classify incoming user queries into one of two categories:\n\n"
        "1. 'CHAT': If the query is a greeting, conversational filler, off-topic chat, "
        " or a generic general-knowledge query that does not require reference to the "
        "ICD-10-CM official guidelines.\n"
        "2. 'RAG': If the query asks for specific rules, guidelines, sequence instructions, "
        "or clinical modification chapters in the ICD-10-CM guidelines (e.g. sepsis, COVID-19,"
        "pregnancy complications, coding conventions). \n\n"
        "Be conservative: if a query has any chance of referencing coding guidelines, classify as RAG."
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{query}")])

   # 4. Chain the prompt and the LLM, then invoke it
    chain = prompt | structured_llm
    decision: RoutingDecision = chain.invoke({"query": query})

    return decision
        