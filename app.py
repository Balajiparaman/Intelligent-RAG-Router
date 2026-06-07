import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="Intelligent RAG Router", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Intelligent RAG Router & Compliance Gatekeeper")
st.markdown(
    "A cost-aware, self-correcting RAG pipeline that filters off-topic queries "
    "and audits LLM responses against official medical guidelines."
)

# API Endpoint URL (pointing to our FastAPI server)
API_URL = "http://127.0.0.1:8000/chat"

# Initialize Session State for Chat History and Sidebar Logs
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = {
        "status": "Ready",
        "intent": "None",
        "reasoning": "Waiting for a query...",
        "is_valid": None,
        "revisions": 0,
        "feedback": ""
    }

# =====================================================================
# Sidebar: System Execution Logs
# =====================================================================
st.sidebar.title("🛠️ System Execution Logs")
st.sidebar.markdown(
    "Observe the middleware's decisions and multi-agent audit loops here in real-time."
)

# Sidebar logs container
log_container = st.sidebar.empty()

def render_sidebar():
    """Renders the current logs in the sidebar."""
    logs = st.session_state.logs
    with log_container.container():
        st.write(f"**Current Status:** `{logs['status']}`")
        st.write(f"**Detected Intent:** `{logs['intent']}`")
        st.write(f"**Router Reasoning:** *{logs['reasoning']}*")
        
        if logs["intent"] == "RAG":
            st.markdown("---")
            st.markdown("### 🔍 Citation Audit Status")
            
            # Show a green checkmark or red cross based on validation
            if logs["is_valid"] is True:
                st.success("✅ CITATION AUDIT PASSED")
            elif logs["is_valid"] is False:
                st.error("❌ CITATION AUDIT REJECTED")
            else:
                st.warning("⏳ Running Audit checks...")
                
            st.write(f"**Revision Loops:** `{logs['revisions']}`")
            if logs["feedback"]:
                st.markdown(f"**Auditor Feedback:**\n> *{logs['feedback']}*")

# Initialize sidebar rendering
render_sidebar()

# =====================================================================
# Main Interface: Chat
# =====================================================================

# Display past messages from chat history
# =====================================================================
# Main Interface: Chat
# =====================================================================

# Display past messages from chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show suggested questions ONLY if chat history is empty
clicked_prompt = None
if not st.session_state.messages:
    st.markdown("### 💡 Suggested Questions")
    st.markdown("Select a question below to test the Cost-Aware RAG Router:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤰 COVID-19 sequencing in pregnancy (RAG / Route to page 71)"):
            clicked_prompt = "What is the specific coding sequencing guideline for a pregnant patient who has COVID-19?"
        if st.button("🩺 General rules for acute vs chronic (RAG / Route to page 14)"):
            clicked_prompt = "What is the guideline for coding acute and chronic conditions?"
    with col2:
        if st.button("🌡️ Sepsis & Septic shock coding rules (RAG / Route to page 23-24)"):
            clicked_prompt = "What is the guideline for coding severe sepsis and septic shock?"
        if st.button("💬 Explain what is medical coding (CHAT / Direct Stream)"):
            clicked_prompt = "Hello! Can you help me learn about medical coding?"

# User Input Box (Checks if user typed OR clicked a suggested question)
prompt = st.chat_input("Ask a medical coding guideline question or chat casually...")
if clicked_prompt:
    prompt = clicked_prompt

if prompt:
    # Display user's message

    # Display user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare logs state for new request
    st.session_state.logs = {
        "status": "Initializing...",
        "intent": "Checking...",
        "reasoning": "Evaluating user query...",
        "is_valid": None,
        "revisions": 0,
        "feedback": ""
    }
    render_sidebar()

    # Display assistant's streaming response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            # Send streaming POST request to FastAPI
            response = requests.post(API_URL, json={"message": prompt}, stream=True)
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        event_data = json.loads(decoded_line[6:])
                        event_type = event_data.get("event")
                        
                        # Handle Status Updates
                        if event_type == "status":
                            st.session_state.logs["status"] = event_data.get("message")
                            render_sidebar()
                            
                        # Handle Routing Decision
                        elif event_type == "route":
                            st.session_state.logs["intent"] = event_data.get("intent")
                            st.session_state.logs["reasoning"] = event_data.get("reasoning")
                            render_sidebar()
                            
                        # Handle Text Token Chunk
                        elif event_type == "token":
                            full_response += event_data.get("text", "")
                            # Render inline typing cursor
                            response_placeholder.markdown(full_response + "▌")
                            
                        # Handle Final Audit Metadata
                        elif event_type == "audit":
                            st.session_state.logs["is_valid"] = event_data.get("is_valid")
                            st.session_state.logs["revisions"] = event_data.get("revisions")
                            st.session_state.logs["feedback"] = event_data.get("feedback")
                            st.session_state.logs["status"] = "Complete"
                            render_sidebar()

            # Remove typing cursor from final text
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error connecting to FastAPI backend: {e}")
