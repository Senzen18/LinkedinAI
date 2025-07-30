import logfire
from typing import List, Optional, TypedDict, Annotated, Literal
import re
import os
import streamlit as st
os.environ['LANGSMITH_OTEL_ENABLED'] = 'true'
os.environ['LANGSMITH_TRACING'] = 'true'

os.environ['LOGFIRE_TOKEN'] = st.secrets["LOGFIRE_TOKEN"]

from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from typing import Dict, Any
import time
from functools import partial
import streamlit as st
# Import the specialised agents
from profile_analyzer_agent import *
from content_gen_agent import *
from career_counsellor_agent import *
from job_matcher_agent import *

openai_api_key = os.getenv("OPENAI_API_KEY")
apify_api_token = os.getenv("APIFY_API_TOKEN")
tavily_api_key = os.getenv("TAVILY_API_KEY")
model_provider = os.getenv("MODEL_PROVIDER", "openai")
gemini_api_key = os.getenv("GEMINI_API_KEY")
print("=====================================================================================os",model_provider,gemini_api_key)
# profile_analyzer_agent = ProfileAnalyzerAgent(openai_api_key=openai_api_key,apify_api_token=apify_api_token).profile_analyzer_agent
# content_gen_agent = ContentGenAgent(openai_api_key=openai_api_key,tavily_api_key=tavily_api_key).content_gen_agent
# career_counsellor_agent = CareerAgent(openai_api_key=openai_api_key).career_counsellor_agent
# job_retrieval_agent = JobRetrievalAgent(openai_api_key=openai_api_key,apify_api_token=apify_api_token).job_retrieval_agent


logfire.configure()
logfire.instrument_pydantic_ai()





def get_llm(temperature=0):
    provider = os.getenv('MODEL_PROVIDER', 'gemini')
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        return ChatOpenAI(model="gpt-4o", temperature=temperature, api_key=api_key)
    elif provider == 'gemini':
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv('GOOGLE_API_KEY')
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature, google_api_key=api_key)
    else:
        raise ValueError("Invalid model provider")
# ------------------------
# 1. Define the shared graph state
# ------------------------

class GraphState(TypedDict):
    """Enhanced state with better memory management."""
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history
    user_profile: Optional[str]           # Scraped or provided LinkedIn profile text
    job_description: Optional[str]        # Target job description (generated or provided)
    agent_response: Optional[str]  
           # Latest agent response (string)
    target_node: Optional[str]      
          # Target node to route to
    profile_url: Optional[str]            # LinkedIn profile URL
    job_role: Optional[str]               # Job role to search for
    profile_analysis: Optional[str]
    # New fields for session management
    user_id: Optional[str]
    session_id: Optional[str]
    agent_context: Optional[str]  # Track which agent is being used
    persistent_data: Optional[Dict[str, Any]]  # For long-term storage


def _extract_lower(text: str) -> str:
    """Utility: safe lower-case conversion."""
    return text.lower() if text else ""


# ------------------------
# 3. Router node
# ------------------------
class ExtractUrlAndRoleOutput(BaseModel):
    profile_url: Optional[str] = Field(description="The LinkedIn profile URL extracted from the user's message.", default=None)
    job_role: Optional[str] = Field(description="The job role extractd from the user's message.", default=None)

def extract_url_and_role(state: GraphState,config: Optional[dict] = None ) -> dict:
    """Extract the LinkedIn profile URL and job role from the user's message."""

    system_prompt = """
    You are a helpful assistant that extracts the LinkedIn profile URL and job role from the user's message.
    
    Rules:
    1. Extract LinkedIn profile URLs that start with 'https://www.linkedin.com/in/' or 'https://www.linkedin.com/'
    2. Extract job roles/titles mentioned in the message. Eg " Senior ML Engineer", "Java Developer", "Data Scientist", "Data Engineer"
    3. If no LinkedIn URL is found, return None for profile_url field.
    4. If no job role is found, return None for the job_role field.
    
    
    """
    user_id = state.get("user_id", "default_user")
    store = "store"
    last_user_message = state["messages"][-1].content
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_message)
    ]

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(ExtractUrlAndRoleOutput)
    result = structured_llm.invoke(messages)
    


    existing_data = load_user_profile_data(user_id, store)
    updates = {}
    if result.profile_url is not None:
        updates["profile_url"] = result.profile_url

    else:
        updates["profile_url"] = existing_data.get("profile_url")
    if result.job_role is not None:
        updates["job_role"] = result.job_role
    else:
        updates["job_role"] = existing_data.get("job_role")
    return updates
    



class RoutingOutput(BaseModel):
    task_type: Literal["profile_analysis", "job_matching", "profile_rewrite", "career_counselling"] = Field(
        description="The type of user request. One of: 'profile_analysis', 'job_matching', 'profile_rewrite', 'career_counselling'"
    )

def Orchestration_Router(state: GraphState) -> dict:
    """
    Orchestration LLM: Classifies the user task into one of four types and routes accordingly.
    - If the user wants to find gaps and inconsistencies in their profile: 'profile_analysis'
    - If the user wants job matching: 'job_matching'
    - If the user wants to rewrite their profile according to a job role: 'profile_rewrite'
    - If the user wants recommendations to upskill based on skill gap: 'career_counselling'
    """
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(RoutingOutput)
    last_user_message = state["messages"][-1].content

    system_prompt = (
        "You are an orchestration agent for a LinkedIn career assistant. "
        "Classify the user's request into one of the following task types:\n"
        "1. profile_analysis: The user wants to find gaps and inconsistencies in their LinkedIn profile.\n"
        "2. job_matching: The user wants to match their profile to jobs or find suitable jobs.\n"
        "3. profile_rewrite: The user wants to rewrite or optimize their profile according to a job role or job description.\n"
        "4. career_counselling: The user wants recommendations to upskill based on the skill gap between their profile and a job description.\n"
        "Respond ONLY with the field 'task_type' and use one of the above values."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_message)
    ]

    routing_result = structured_llm.invoke(messages)
    task_type = routing_result.task_type

    # Map task_type to node names in the graph
    if task_type == "profile_analysis":
        return {"target_node": "profile_analyzer"}
    elif task_type == "job_matching":
        return {"target_node": "job_matcher"}
    elif task_type == "profile_rewrite":
        return {"target_node": "content_generator"}
    elif task_type == "career_counselling":
        return {"target_node": "career_counsellor"}
    else:
        # Default fallback
        return {"target_node": "profile_analyzer"}

# ------------------------
# 4. Agent wrapper nodes
# ------------------------


def create_memory_stores():
    """Create both checkpoint and store for comprehensive memory."""
    
    # Short-term memory (conversation within thread)
    checkpointer = InMemorySaver()  # Use PostgresSaver for production
    
    # Long-term memory (across sessions and threads)
    store = InMemoryStore()  # Use PostgresStore for production
    
    return checkpointer, store

def save_user_profile_data(user_id: str, profile_data: dict, store):
    """Save user profile data to long-term memory."""
    namespace = (user_id, "profile_data")
    
    # Save profile information
    if profile_data.get("profile_url"):
        store.put(namespace, "profile_url", {"url": profile_data["profile_url"]})
    
    if profile_data.get("user_profile"):
        store.put(namespace, "user_profile", {"content": profile_data["user_profile"]})
    
    if profile_data.get("job_role"):
        store.put(namespace, "job_role", {"role": profile_data["job_role"]})
    
    if profile_data.get("job_description"):
        store.put(namespace, "job_description", {"content": profile_data["job_description"]})

def load_user_profile_data(user_id: str, store) -> dict:
    """Load user profile data from long-term memory."""
    namespace = (user_id, "profile_data")
    profile_data = {}
    
    try:
        # Retrieve stored data
        profile_url_item = store.get(namespace, "profile_url")
        if profile_url_item:
            profile_data["profile_url"] = profile_url_item.value["url"]
            
        user_profile_item = store.get(namespace, "user_profile")
        if user_profile_item:
            profile_data["user_profile"] = user_profile_item.value["content"]
            
        job_role_item = store.get(namespace, "job_role")
        if job_role_item:
            profile_data["job_role"] = job_role_item.value["role"]
            
        job_description_item = store.get(namespace, "job_description")
        if job_description_item:
            profile_data["job_description"] = job_description_item.value["content"]
            
    except Exception as e:
        print(f"Error loading user profile data: {e}")
    
    return profile_data

def save_agent_conversation(user_id: str, agent_name: str, conversation_data: dict, store):
    """Save agent-specific conversation history."""
    namespace = (user_id, f"agent_conversations_{agent_name}")
    
    # Create a unique key based on timestamp or conversation ID
    conversation_key = f"conv_{int(time.time())}"
    
    store.put(namespace, conversation_key, {
        "timestamp": time.time(),
        "messages": conversation_data.get("messages", []),
        "agent_response": conversation_data.get("agent_response"),
        "context": conversation_data.get("context", {})
    })

def load_agent_conversation_history(user_id: str, agent_name: str, store, limit: int = 5):
    """Load recent conversation history for a specific agent."""
    namespace = (user_id, f"agent_conversations_{agent_name}")
    
    try:
        # Search for recent conversations
        items = store.search(namespace, limit=limit)
        
        # Sort by timestamp (most recent first)
        sorted_items = sorted(items, key=lambda x: x.value.get("timestamp", 0), reverse=True)
        
        return [item.value for item in sorted_items]
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return []

def load_cached_profile(profile_url: str, store) -> Optional[str]:
    if not profile_url:
        return None
    namespace = ("linkedin_profiles",)
    item = store.get(namespace, profile_url)
    if item:
        return item.value.get("content")
    return None

def save_cached_profile(profile_url: str, user_profile: str, store):
    if not profile_url or not user_profile:
        return
    namespace = ("linkedin_profiles",)
    store.put(namespace, profile_url, {"content": user_profile})

def load_cached_job_description(job_role: str, store) -> Optional[str]:
    if not job_role:
        return None
    key = job_role.lower()
    namespace = ("job_descriptions",)
    item = store.get(namespace, key)
    if item:
        return item.value.get("content")
    return None

def save_cached_job_description(job_role: str, job_description: str, store):
    if not job_role or not job_description:
        return
    key = job_role.lower()
    namespace = ("job_descriptions",)
    store.put(namespace, key, {"content": job_description})

def profile_analyzer_node(state: GraphState, config: Optional[dict] = None) -> dict:
    """Enhanced profile analyzer with memory management."""

    profile_analyzer_agent = ProfileAnalyzerAgent(openai_api_key=openai_api_key,apify_api_token=apify_api_token, model_provider=model_provider, gemini_api_key=gemini_api_key).profile_analyzer_agent
    user_id = state.get("user_id", "default_user")
    store = config["configurable"]["store"]
    # Load existing profile data
    existing_data = load_user_profile_data(user_id, store)
    
    # Use existing profile if available and no new URL provided
    if existing_data.get("user_profile") and not state.get("profile_url"):
        state["user_profile"] = existing_data["user_profile"]
        state["profile_url"] = existing_data.get("profile_url")
    
    last_user_message = state["messages"][-1].content
    
    if state.get("user_profile"):
        prompt = f"{last_user_message}\n\nProfile text: {state['user_profile']}"
    else:
        prompt = last_user_message
    
    result = profile_analyzer_agent.run_sync(prompt)
    response_text = result.output if hasattr(result, "output") else str(result)
    
    # Save conversation history for this agent
    save_agent_conversation(user_id, "profile_analyzer", {
        "messages": state["messages"],
        "agent_response": response_text,
        "context": {"profile_url": state.get("profile_url")}
    }, store)
    
    # Update persistent data
    profile_data = {
        "profile_url": state.get("profile_url"),
        "user_profile": state.get("user_profile"),
        "profile_analysis": response_text
    }
    save_user_profile_data(user_id, profile_data, store)
    
    return {
        "profile_analysis": response_text,
        "agent_response": response_text,
        "messages": [AIMessage(content=response_text)],
        "agent_context": "profile_analyzer"
    }

def content_generator_node(state: GraphState, config: Optional[dict] = None) -> dict:
    """Enhanced content generator with memory management."""

    content_gen_agent = ContentGenAgent(openai_api_key=openai_api_key,tavily_api_key=tavily_api_key, model_provider=model_provider, gemini_api_key=gemini_api_key).content_gen_agent
    user_id = state.get("user_id", "default_user")
    config = config.get("configurable", {})
    store = config.get("store", "store")
    # Load existing data
    existing_data = load_user_profile_data(user_id, store)
    
    # Use existing data if not in current state
    if not state.get("user_profile") and existing_data.get("user_profile"):
        state["user_profile"] = existing_data["user_profile"]
    
    if not state.get("job_description") and existing_data.get("job_description"):
        state["job_description"] = existing_data["job_description"]
    
    # Load conversation history for context
    conversation_history = load_agent_conversation_history(user_id, "content_generator", store)
    
    # Existing logic
    user_profile = state.get("user_profile")
    job_desc = state.get("job_description")

    if user_profile and job_desc:
        prompt = f"""
        User Profile:
        {user_profile}
        Job Description:
        {job_desc}
        """
    else:
        prompt = state["messages"][-1].content

    result = content_gen_agent.run_sync(prompt)
    response_text = result.output if hasattr(result, "output") else str(result)
    
    # Save conversation history
    save_agent_conversation(user_id, "content_generator", {
        "messages": state["messages"],
        "agent_response": response_text,
        "context": {"job_description": state.get("job_description")}
    }, store)
    
    # Update persistent data
    profile_data = {
        "user_profile": state.get("user_profile"),
        "job_description": state.get("job_description")
    }
    save_user_profile_data(user_id, profile_data, store)

    return {
        "agent_response": response_text,
        "messages": [AIMessage(content=response_text)],
        "agent_context": "content_generator"
    }

def career_counsellor_node(state: GraphState, config: Optional[dict] = None) -> dict:
    """Enhanced career counsellor with memory management."""

    career_counsellor_agent = CareerAgent(openai_api_key=openai_api_key,apify_api_token=apify_api_token, model_provider=model_provider, gemini_api_key=gemini_api_key).career_counsellor_agent
    user_id = state.get("user_id", "default_user")
    store = config["configurable"]["store"]
    # Load existing data
    existing_data = load_user_profile_data(user_id, store)
    
    # Use existing data if not in current state
    if not state.get("user_profile") and existing_data.get("user_profile"):
        state["user_profile"] = existing_data["user_profile"]
    
    if not state.get("job_description") and existing_data.get("job_description"):
        state["job_description"] = existing_data["job_description"]
    
    # Load conversation history for context
    conversation_history = load_agent_conversation_history(user_id, "career_counsellor", store)
    
    # Existing logic
    user_profile = state.get("user_profile")
    job_desc = state.get("job_description")

    if user_profile and job_desc:
        prompt = f"""
        User Profile:
        {user_profile}
        Job Description:
        {job_desc}
        """
    else:
        prompt = state["messages"][-1].content

    result = career_counsellor_agent.run_sync(prompt)
    response_text = result.output if hasattr(result, "output") else str(result)
    
    # Save conversation history
    save_agent_conversation(user_id, "career_counsellor", {
        "messages": state["messages"],
        "agent_response": response_text,
        "context": {"job_description": state.get("job_description")}
    }, store)
    
    # Update persistent data
    profile_data = {
        "user_profile": state.get("user_profile"),
        "job_description": state.get("job_description")
    }
    save_user_profile_data(user_id, profile_data, store)

    return {
        "agent_response": response_text,
        "messages": [AIMessage(content=response_text)],
        "agent_context": "career_counsellor"
    }


def _entry_gate(state: GraphState) -> str:
    """
    Pure routing function - determines which node to go to next based on state.
    Does NOT update the state, only returns the next node name.
    """
    target = state.get("target_node")
    
    # If no target is set, default to profile analyzer
    if not target:
        return "profile_analyzer"
    
    # Check if we need a profile for all agents including analyzer
    if not state.get("user_profile"):
        needs_profile = target in ["profile_analyzer", "job_matcher", "content_generator", "career_counsellor"]
        if needs_profile:
            return "scrape_linkedin_profile"
    
    # Check if we need a job description for other agents
    if not state.get("job_description"):
        needs_job = target in ["job_matcher", "content_generator", "career_counsellor"]
        if needs_job:
            return "job_retriever"
    
    # All prerequisites met, route to the target agent
    return target

def entry_gate(state: GraphState) -> dict:
    return {"__next__": _entry_gate(state)}

# The job matcher agent in job_matcher_agent.py uses a LangGraph node function already (job_matcher_node)
# We'll wrap it to ensure state compatibility.

def job_matcher_wrapper_node(state: GraphState, config: Optional[dict] = None) -> dict:
    """Enhanced job matcher with memory management."""
    user_id = state.get("user_id", "default_user")
    store = config["configurable"]["store"]
    # Load existing data
    existing_data = load_user_profile_data(user_id, store)
    
    # Use existing data if not in current state
    if not state.get("user_profile") and existing_data.get("user_profile"):
        state["user_profile"] = existing_data["user_profile"]
    
    if not state.get("job_description") and existing_data.get("job_description"):
        state["job_description"] = existing_data["job_description"]
    
    # Load conversation history for context
    conversation_history = load_agent_conversation_history(user_id, "job_matcher", store)
    
    # Run job matcher
    jm_state = {
        "user_profile": state.get("user_profile"),
        "job_description": state.get("job_description"),
    }
    
    updated = job_matcher_node(jm_state)
    match_report = updated.get("match_report")
    agent_response = updated.get("agent_response")
    
    # Save conversation history
    save_agent_conversation(user_id, "job_matcher", {
        "messages": state["messages"],
        "agent_response": str(agent_response),
        "context": {
            "job_description": state.get("job_description"),
            "match_report": str(match_report)
        }
    }, store)
    
    # Update persistent data
    profile_data = {
        "user_profile": state.get("user_profile"),
        "job_description": state.get("job_description")
    }
    save_user_profile_data(user_id, profile_data, store)
    
    return {
        "agent_response": str(match_report),
        "messages": [AIMessage(content=str(agent_response))],
        "agent_context": "job_matcher"
    }

def job_retriever_node(state: GraphState, config: Optional[dict] = None) -> dict:
    """Enhanced job retriever with memory management."""

    job_retrieval_agent = JobRetrievalAgent(openai_api_key=openai_api_key,apify_api_token=apify_api_token, model_provider=model_provider, gemini_api_key=gemini_api_key).job_retrieval_agent
    user_id = state.get("user_id", "default_user")
    store = config["configurable"]["store"]
    existing_data = load_user_profile_data(user_id, store)
    job_description = None
    job_role = state.get("job_role")
    if not state.get("job_role") or state.get("job_role") == "null":
        if existing_data.get("job_role") == None:
            job_role = interrupt(" Before we can match your profile to jobs, please provide the job role you are interested in.")

        else:
            job_role = existing_data.get("job_role")
            job_description = existing_data.get("job_description")
        
    if not job_role:
        return {
            "agent_response": "Please provide the job role you are interested in.",
            "messages": [AIMessage(content="Please provide the job role you are interested in.")]
        }
    cached_job_description = load_cached_job_description(job_role, store)
    if cached_job_description:
        job_description = cached_job_description
  
    if not job_description:
        try:
            result = job_retrieval_agent.run_sync(f"Generate a job description for: {job_role}")
            job_description = result.output
        except Exception as e:
            return {
                "agent_response": f"Error retrieving job description: {str(e)}",
                "messages": [AIMessage(content=f"Error retrieving job description: {str(e)}")]
            }
        profile_data = {
            "job_role": job_role,
            "job_description": job_description
        }
        save_user_profile_data(user_id, profile_data, store)
    save_cached_job_description(job_role, job_description, store)
    return {
        "job_description": job_description,
        "job_role": job_role,
        "messages": [SystemMessage(content=f"Successfully retrieved job description for {job_role}")]
    }

def scrape_linkedin_profile_node(state: GraphState, config: Optional[dict] = None) -> dict:
    """Enhanced scrape LinkedIn profile with memory management."""
    user_id = state.get("user_id", "default_user")
    store = config["configurable"]["store"]
    existing_data = load_user_profile_data(user_id, store)
    user_profile = None
    profile_url = state.get("profile_url")
    if state.get("profile_url") == None or state.get("profile_url") == "null":
        if existing_data.get("profile_url") == None:
            profile_url = interrupt(" Before we can match your profile to jobs, please provide your LinkedIn profile URL.")

        else:
            profile_url = existing_data.get("profile_url",None)
            user_profile = existing_data.get("user_profile",None)


    if not profile_url:
        return {
            "agent_response": "Please provide your LinkedIn profile URL.",
            "messages": [AIMessage(content="Please provide your LinkedIn profile URL.")]
        }
    
    cached_profile = load_cached_profile(profile_url, store)
    if cached_profile:
        user_profile = cached_profile
    
        
    if not user_profile :
        try:
            result = scrape_linkedin_profile_plain(profile_url)
            user_profile = result.profile_text
        except Exception as e:
            return {
                "agent_response": f"Error scraping profile: {str(e)}",
                "messages": [AIMessage(content=f"Error scraping profile: {str(e)}")]
            }
        # Save to persistent data
        profile_data = {
            "profile_url": profile_url,
            "user_profile": user_profile
        }
        save_user_profile_data(user_id, profile_data, store)
    save_cached_profile(profile_url, user_profile, store)
    return {
        "user_profile": user_profile,
        "messages": [SystemMessage(content=f"Successfully scraped profile from {profile_url}")]
    }
    


    # Check cache first
    # cached_profile = load_cached_profile(profile_url, store)

 

def finish_conversation(state: GraphState):
    last_message = state["agent_response"]
    writer = get_stream_writer()
    writer(str(last_message))
    return {}

# ------------------------
# 5. Build the LangGraph
# ------------------------

builder = StateGraph(GraphState)

# Create checkpointer and store before builder
checkpointer, store = create_memory_stores()

# Add core nodes
builder.add_node("extract_url_and_role", extract_url_and_role)
builder.add_node("Orchestration_Router", Orchestration_Router)
builder.add_node("profile_analyzer", profile_analyzer_node)
builder.add_node("content_generator", content_generator_node)
builder.add_node("career_counsellor", career_counsellor_node)
builder.add_node("job_matcher", job_matcher_wrapper_node)
builder.add_node("entry_gate", entry_gate)
builder.add_node("job_retriever", job_retriever_node)
builder.add_node("scrape_linkedin_profile", scrape_linkedin_profile_node)
builder.add_node("finish_conversation", finish_conversation)

# Conditional branching from router

builder.add_edge("extract_url_and_role", "Orchestration_Router")
builder.add_edge("Orchestration_Router", "entry_gate")

builder.add_conditional_edges(
    "entry_gate",
    lambda s:s["__next__"],
    {
        "profile_analyzer": "profile_analyzer",
        "job_retriever": "job_retriever",
        "job_matcher": "job_matcher",
        "content_generator": "content_generator",
        "career_counsellor": "career_counsellor",
        "scrape_linkedin_profile": "scrape_linkedin_profile",
    }
)

# Data gathering nodes loop back to entry_gate
builder.add_edge("scrape_linkedin_profile", "entry_gate")
builder.add_edge("job_retriever", "entry_gate")

# Final agent nodes go to finish_conversation
for node_name in [
    "profile_analyzer",
    "content_generator",
    "career_counsellor",
    "job_matcher",
]:
    builder.add_edge(node_name, "finish_conversation")

builder.add_edge("finish_conversation", END)
# Set entry point
builder.set_entry_point("extract_url_and_role")

# Compile the graph (remove store from compile if not supported, assuming it's custom)
multi_agent_graph = builder.compile(checkpointer=checkpointer)

mermaid_syntax = multi_agent_graph.get_graph().draw_mermaid()

# Save as PNG file
png_bytes = draw_mermaid_png(
    mermaid_syntax=mermaid_syntax,
    output_file_path="my_graph.png",  # Specify the file path
    background_color="white",
    padding=10
)
# ------------------------
# 6. Convenience run function
# ------------------------
from langgraph.types import Command

def run_query(query: str, user_id: str = "default_user", thread_id=1):
    """Helper to run a single turn through the graph with interrupt handling and memory loading."""
    
    # Load existing profile data for the user
    existing_data = load_user_profile_data(user_id, store)
    
    initial_state: GraphState = {
        "messages": [HumanMessage(content=query)],
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
    
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "store": store}}
    
    # Initial graph invocation
    final_state = multi_agent_graph.invoke(initial_state, config=config)
    
    # Handle interrupts
    while "__interrupt__" in final_state:
        interrupt_data = final_state["__interrupt__"]
        
        # Extract message from interrupt data
        message = extract_interrupt_message(interrupt_data)
        
        # Get user input
        user_input = ""
        while not user_input.strip():
            user_input = input(f"{message}\n> ").strip()
            if not user_input:
                print("Please provide a valid input.")
        
        # Resume with user input
        final_state = multi_agent_graph.invoke(
            Command(resume=user_input),
            config=config
        )
    
    return final_state.get("agent_response")

def extract_interrupt_message(interrupt_data):
    """Helper function to extract message from various interrupt formats."""
    if isinstance(interrupt_data, list):
        # Handle list of interrupts
        if len(interrupt_data) > 0:
            interrupt_obj = interrupt_data[0]
            if hasattr(interrupt_obj, 'value'):
                return interrupt_obj.value.strip()
            else:
                return str(interrupt_obj).strip()
    elif hasattr(interrupt_data, 'value'):
        # Handle single Interrupt object
        return interrupt_data.value.strip()
    else:
        # Handle string or other types
        return str(interrupt_data).strip()
    
    return "Please provide input:"



if __name__ == "__main__":
    # Simple CLI loop for manual testing
    print("\n🤖 Multi-Agent Career Assistant. Type 'exit' to quit.\n")
    cached_profile_url: Optional[str] = "https://www.linkedin.com/in/sendhan-a-38a48811b/"
    cached_job_role: Optional[str] = "Senior Machine Learning Engineer"

    thread_counter = 1  # Increment thread_id for each query to simulate new sessions

    while True:
        user_inp = input("You: ")
        if user_inp.lower() in {"exit", "quit"}:
            break

        response = run_query(user_inp, user_id="default_user", thread_id=thread_counter)
        thread_counter += 1

        print(f"Agent:\n{response}\n")
