# Cost-Aware RAG Router & Compliance Gatekeeper

An enterprise-grade asynchronous AI infrastructure proxy built with **FastAPI**, **LangGraph**, and **ChromaDB**. 

This middleware optimizes large language model (LLM) pipelines by dynamically routing casual conversation to a cheap direct-streaming endpoint while processing complex, domain-specific queries through an advanced **Parent-Child RAG pipeline** paired with a stateful, self-correcting multi-agent citation audit loop.

---

## 🏗️ Architecture Overview

```
                        [ User Query ]
                              │
                              ▼
                ┌───────────────────────────┐
                │   FastAPI Web Gateway     │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌───────────────────────────┐
                │    Intent Router Node     │
                └─────────────┬─────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼ (CHAT Intent)                 ▼ (RAG Intent)
       ┌──────────────┐             ┌─────────────────────────────────┐
       │ Direct LLM   │             │   Stateful LangGraph Pipeline   │
       │ Token Stream │             │  - Parent-Child ChromaDB Search │
       └──────────────┘             │  - Self-Correcting Citation Loop│
                                    └─────────────────────────────────┘
```

1. **Intent-Based Cost Routing**: Incoming queries are processed by a lightweight router. Casual queries (greetings, off-topic chat) bypass the database completely, saving on database latency and token costs.
2. **Parent-Child Retrieval Strategy**: Complex queries trigger a structural search in ChromaDB. Small semantic sentences (**children**) are searched for maximum precision, but we retrieve and feed the broader surrounding pages (**parents**) into the model to preserve complete context.
3. **Stateful Self-Correcting Auditor Loop**: The **Researcher Agent** synthesizes answers with strict inline citations. The **Auditor Agent** validates these claims against the raw page text. If a hallucination or incorrect page citation is detected, the loop automatically reroutes back to the Researcher for correction before serving.

---

## 📂 Project Directory Structure

```text
cost_aware_rag/
│
├── main.py                 # FastAPI Application layer & SSE streaming endpoints
├── graph.py                # LangGraph node configurations, state definition, and routing
├── database.py             # ChromaDB vector store ingestion & parent-child retrieval
├── router.py               # Intent-based structured output routing logic
│
├── data/
│   └── medical_data.pdf    # Unstructured medical coding guidelines (ICD-10-CM)
│
├── chroma_db/              # Persisted local database files (generated on ingestion)
└── requirements.txt        # System library requirements

app.py                      # Streamlit chat interface with real-time execution logs
test_api.py                 # CLI client to test FastAPI Server-Sent Events (SSE)
```

---

## 🚀 Key Features

* **Asynchronous SSE Streaming**: Streams responses token-by-token using FastAPI's `StreamingResponse` alongside live agent state updates.
* **100% Free & Local Stack**: Employs open-source Sentence-Transformers (`all-MiniLM-L6-v2`) for local vector calculations and ChromaDB for local file persistence.
* **Structured Output Guarantees**: Leverages Pydantic schemas to enforce strict JSON outputs from the Intent Router and Auditor agents.
* **Streamlit UI Logging Dashboard**: Displays a split-pane interface showing the chat window on the left, and live backend execution metrics (intent, reasoning, audit status, loops run) on the right.

---

## 🛠️ Setup & Installation Instructions

Follow these steps to run the complete stack locally on your machine:

### 1. Configure the Environment
Ensure you have a Python virtual environment activated, and install the required libraries:

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r cost_aware_rag/requirements.txt
```

### 2. Add API Keys
Create a `.env` file at the root of the directory:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key
```

### 3. Ingest the Guidelines Database
Before running the server, parse and ingest the PDF into your local vector database:
```bash
python cost_aware_rag/test_db.py
```
*This extracts text page-by-page, generates 1,992 child chunks, embeds them, and saves the database inside `cost_aware_rag/chroma_db/`.*

### 4. Start the FastAPI Backend
Start the high-performance ASGI server:
```bash
uvicorn cost_aware_rag.main:app --reload
```
*The server will run on `http://127.0.0.1:8000`.*

### 5. Run the Streamlit Interface
In a new terminal window (with the virtual environment activated), start the user interface:
```bash
streamlit run app.py
```
*This will open the web interface in your default browser at `http://localhost:8501`.*

---

## 📋 Verification Examples

### Chat Request (Direct Path)
* **User Input**: `"Hello! Can you help me learn about medical coding?"`
* **Router Action**: Detects `CHAT` intent.
* **Execution Logs**: 
  * *Intent*: `CHAT`
  * *Reasoning*: Query is a general greeting and conversational.
  * *Performance*: Zero database queries, immediate token-by-token streaming response.

### RAG Request (Stateful Agent Path)
* **User Input**: `"What is the coding sequencing guideline for a pregnant patient with COVID-19?"`
* **Router Action**: Detects `RAG` intent.
* **Execution Logs**:
  * *Intent*: `RAG`
  * *Database Action*: ChromaDB retrieves Page 71.
  * *Auditor Action*: Scans Researcher's text, confirms references match Page 71 text, and returns `is_valid: True` inside the audit report.
  * *Revision Loops*: `1` (Passed first validation attempt).