
import streamlit as st
import os
 # Assuming multi_agent_main.py is in the same directory
import openai
from apify_client import ApifyClient
from tavily import TavilyClient
from openai import OpenAI, AuthenticationError as OpenAIAuthError
from langgraph.types import Command
# from multi_agent_main import multi_agent_graph, load_user_profile_data, store, extract_interrupt_message
import logfire
import google.generativeai as genai
os.environ['LOGFIRE_TOKEN'] = st.secrets["LOGFIRE_TOKEN"]
# Page configuration
st.set_page_config(
    page_title="LinkedIn AI Career Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for eye-catching UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 1px solid #0077b5;
        color: #000000 !important;
    }
    .stButton > button {
        background-color: #0077b5;
        color: white;
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        max-width: 700px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    .chat-message.user {
        background-color: #0077b5;
        color: #ffffff;
        text-align: right;
    }
    .chat-message.assistant {
        background-color: #2f2f2f;
        color: #ffffff;
    }
    .stSpinner {
        color: #0077b5;
    }
    /* Center main content and reduce width */
    .block-container{
        max-width: 900px;
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'openai': '',
        'apify': '',
        'tavily': '',
        'gemini': ''
    }
if 'validated' not in st.session_state:
    st.session_state.validated = {
        'model': False,
        'apify': False,
        'tavily': False
    }
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = 1  # Fixed for session, increment if needed for new convos

# Functions to update keys in real-time
def update_openai_key():
    st.session_state.api_keys['openai'] = st.session_state.openai_input

def update_apify_key():
    st.session_state.api_keys['apify'] = st.session_state.apify_input

def update_tavily_key():
    st.session_state.api_keys['tavily'] = st.session_state.tavily_input

def update_gemini_key():
    st.session_state.api_keys['gemini'] = st.session_state.gemini_input

# Validation functions
def validate_openai_key(api_key: str) -> bool:
    if not api_key:
        return False
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except OpenAIAuthError:
        return False
    except Exception:
        return False

def validate_apify_key(api_key: str) -> bool:
    if not api_key:
        return False
    try:
        client = ApifyClient(api_key)
        user_info = client.user().get()
        return bool(user_info)
    except Exception:
        return False

def validate_tavily_key(api_key: str) -> bool:
    if not api_key:
        return False
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search("test query", max_results=1)
        return bool(response)
    except Exception:
        return False

def validate_gemini_key(api_key: str) -> bool:
    if not api_key:
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("test")
        print("Response",response)
        return True
    except Exception as e:
        print("Error validating Gemini API key",e)  
        return False

# Helper to build initial graph state

def build_initial_state(user_query: str, user_id: str, thread_id: int):
    existing_data = load_user_profile_data(user_id, store)
    return {
        "messages": [multi_agent_main.HumanMessage(content=user_query)],
        "user_profile": None,
        "job_description": None,
        "agent_response": None,
        "target_node": None,
        "profile_url": None,
        "job_role": None,
        "profile_analysis": None,
        "user_id": user_id,
        "session_id": str(thread_id),
        "agent_context": None,
        "persistent_data": existing_data,
    }

# Sidebar for API keys
with st.sidebar:
    st.title("🔑 API Keys")
    st.markdown("Enter your API keys below to start chatting.")
    
    input_type = "password"
    
    st.selectbox("LLM Provider", ["OpenAI", "Gemini"], key="model_provider")
    provider = st.session_state.model_provider
    
    if provider == "OpenAI":
        st.text_input(
            "OpenAI API Key",
            type=input_type,
            value=st.session_state.api_keys['openai'],
            key="openai_input",
            on_change=update_openai_key
        )
    else:
        st.text_input(
            "Gemini API Key",
            type=input_type,
            value=st.session_state.api_keys['gemini'],
            key="gemini_input",
            on_change=update_gemini_key
        )
    
    st.text_input(
        "Apify API Key",
        type=input_type,
        value=st.session_state.api_keys['apify'],
        key="apify_input",
        on_change=update_apify_key
    )
    st.text_input(
        "Tavily API Key",
        type=input_type,
        value=st.session_state.api_keys['tavily'],
        key="tavily_input",
        on_change=update_tavily_key
    )
    
    if st.button("Proceed"):
        # Validate keys
        provider = st.session_state.model_provider
        if provider == "OpenAI":
            st.session_state.validated['model'] = validate_openai_key(st.session_state.api_keys['openai'])
        else:
            st.session_state.validated['model'] = validate_gemini_key(st.session_state.api_keys['gemini'])
        
        st.session_state.validated['apify'] = validate_apify_key(st.session_state.api_keys['apify'])
        st.session_state.validated['tavily'] = validate_tavily_key(st.session_state.api_keys['tavily'])
        
        if st.session_state.validated['model'] and st.session_state.validated['apify'] and st.session_state.validated['tavily']:
            st.success("All API keys validated successfully!")
        else:
            if not st.session_state.validated['model']:
                st.error(f"Invalid {provider} API key")
            if not st.session_state.validated['apify']:
                st.error("Invalid Apify API key")
            if not st.session_state.validated['tavily']:
                st.error("Invalid Tavily API key")

# Set environment variables if keys are provided and validated
if st.session_state.validated['model']:
    provider = st.session_state.model_provider
    os.environ['MODEL_PROVIDER'] = provider.lower()
    if provider == "OpenAI":
        os.environ['OPENAI_API_KEY'] = st.session_state.api_keys['openai']
        os.environ['MODEL_API_KEY'] = st.session_state.api_keys['openai']
    else:
        os.environ['GEMINI_API_KEY'] = st.session_state.api_keys['gemini']
        os.environ['GOOGLE_API_KEY'] = st.session_state.api_keys['gemini']
        os.environ['MODEL_API_KEY'] = st.session_state.api_keys['gemini']
if st.session_state.validated['apify']:
    os.environ['APIFY_API_TOKEN'] = st.session_state.api_keys['apify']
if st.session_state.validated['tavily']:
    os.environ['TAVILY_API_KEY'] = st.session_state.api_keys['tavily']

# Main chat interface
st.title("🤖 LinkedIn AI Career Assistant")
st.markdown("Chat with our AI to get career advice, profile analysis, job matching, and more!")

# Check if all keys are provided and validated
all_keys_valid = st.session_state.validated['model'] and st.session_state.validated['apify'] and st.session_state.validated['tavily']

if not all_keys_valid:
    st.warning("Please enter and validate all API keys in the sidebar before starting the chat.")
else:
    # Display chat history
    import multi_agent_main
    from multi_agent_main import (
        multi_agent_graph, load_user_profile_data, store, extract_interrupt_message
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # If there's no pending interrupt, treat as new query
        if not st.session_state.get("pending_interrupt"):
            # Add user message to history and UI
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build initial state and config
            initial_state = build_initial_state(prompt, "default_user", st.session_state.thread_id)
            config = {"configurable": {"thread_id": f"default_user_{st.session_state.thread_id}", "store": store}}

            # Invoke graph
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result_state = multi_agent_graph.invoke(initial_state, config=config)

            # Handle interrupt or normal response
            if "__interrupt__" in result_state:
                interrupt_msg = extract_interrupt_message(result_state["__interrupt__"])
                st.session_state.pending_interrupt = True
                st.session_state.pending_config = config
                st.write(interrupt_msg)
                st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
            else:
                response = result_state.get("agent_response", "")
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.thread_id += 1  # Increment for next new query
        else:
            # We are answering an interrupt prompt
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Resume graph
            config = st.session_state.pending_config
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result_state = multi_agent_graph.invoke(Command(resume=prompt), config=config)

            if "__interrupt__" in result_state:
                interrupt_msg = extract_interrupt_message(result_state["__interrupt__"])
                st.session_state.pending_interrupt = True
                st.write(interrupt_msg)
                st.session_state.messages.append({"role": "assistant", "content": interrupt_msg})
            else:
                response = result_state.get("agent_response", "")
                st.session_state.pending_interrupt = False
                st.session_state.pending_config = None
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.thread_id += 1
