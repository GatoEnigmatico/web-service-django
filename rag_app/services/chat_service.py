from typing import Dict, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# We do not use ToolNode in this case, as the flow is linear

from .vector_service import faiss_manager
from .csv_storage import get_stored_answer, store_answer


# Helper functions
def retrieve_context(user_prompt: str) -> str:
    """Retrieves context using the vector store."""
    relevant_docs = faiss_manager.search(user_prompt)
    context_parts = []
    for doc in relevant_docs:
        if isinstance(doc, str):
            context_parts.append(doc)
        elif isinstance(doc, dict) and "content" in doc:
            context_parts.append(doc["content"])
        else:
            context_parts.append(str(doc))
    return "\n".join(context_parts)


def build_conversation_history(
    history: List[dict], user_prompt: str, system_instructions: str
) -> List:
    """Assembles the conversation history to send to the model."""
    messages = [SystemMessage(content=system_instructions)]
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_prompt))
    return messages


def generate_ai_response(messages: List) -> str:
    """Generates the AI response using ChatOpenAI."""
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    ai_response = chat_model(messages)
    return ai_response.content


def update_history(
    history: List[dict], user_prompt: str, ai_response: str
) -> List[dict]:
    """Updates the conversation history with the new exchange."""
    history.append({"role": "user", "content": user_prompt})
    history.append({"role": "assistant", "content": ai_response})
    return history


# Node functions for StateGraph
def node_build_history(state: Dict) -> Dict:
    history = state.get("history", [])
    user_prompt = state["user_prompt"]
    system_instructions = state["system_instructions"]
    messages = build_conversation_history(history, user_prompt, system_instructions)
    state["messages"] = messages
    return state


def node_generate_response(state: Dict) -> Dict:
    messages = state["messages"]
    ai_response = generate_ai_response(messages)
    state["ai_response"] = ai_response
    return state


def node_update_history(state: Dict) -> Dict:
    history = state.get("history", [])
    user_prompt = state["user_prompt"]
    ai_response = state["ai_response"]
    updated_history = update_history(history, user_prompt, ai_response)
    state["updated_history"] = updated_history
    return state


# Function to signal the graph to end the flow
def always_end(state: Dict) -> str:
    return END


def handle_chat(user_prompt: str, history: List[dict], thread_id: str, system_instructions: str):
    """
    Handles conversation using StateGraph with persistence based on `thread_id`.
    """
    # Create the StateGraph and add nodes sequentially
    workflow = StateGraph(dict)  # Using dict for state
    workflow.add_node("build_history", node_build_history)
    workflow.add_node("generate_response", node_generate_response)
    workflow.add_node("update_history", node_update_history)

    # Sequential connections
    workflow.add_edge(START, "build_history")
    workflow.add_edge("build_history", "generate_response")
    workflow.add_edge("generate_response", "update_history")
    workflow.add_conditional_edges("update_history", always_end)

    # Initialize MemorySaver without `configurable`
    checkpointer = MemorySaver()

    # Define the initial state
    state = {
        "user_prompt": user_prompt,
        "history": history,
        "system_instructions": system_instructions
    }

    # Compile the LangGraph workflow
    app = workflow.compile(checkpointer=checkpointer)

    # Define configuration with `configurable`
    config = {"configurable": {"thread_id": thread_id}}

    # Invoke the graph with configuration
    final_state = app.invoke(state, config)

    ai_response = final_state.get("ai_response", "")
    updated_history = final_state.get("updated_history", history)

    return ai_response, updated_history
