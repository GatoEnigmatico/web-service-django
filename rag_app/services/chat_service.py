import datetime

from typing_extensions import TypedDict
from django.conf import settings
from pathlib import Path
import json
from typing import Annotated, Optional, List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages.utils import trim_messages
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


from .prompts_service import (
    PROP_INICIALIZAR_CONVERSACION,
    PROP_REALIZAR_PREGUNTA,
    PROP_RESPONDER_AL_USUARIO,
    build_prompt,
    build_system_prompt,
)
from .questions import QUESTIONS

# Construct the path relative to the Django project
PROMPTS_PATH = Path(settings.BASE_DIR) / "config" / "prompts.json"


# Load prompts from the JSON file
def load_prompts() -> dict:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# Usage
PROMPTS = load_prompts()


class Question(TypedDict):
    key: str
    question: str
    answer: Optional[str]  # Respuesta opcional, por defecto None


class State(MessagesState):
    messages: Annotated[list, add_messages]
    questions: List[Question]  # Lista de preguntas con respuestas opcionales


# Inicializar MemorySaver
memory = MemorySaver()


@tool
def search(query: str):
    """Check if the question anwer a question"""
    return "Se respondio una pregunta?"


tools = [search]

tool_node = ToolNode(tools)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)


prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(build_system_prompt()),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

runnable = llm


def agent(state: State) -> State:

    user_prompt = state["messages"][-1].content if state["messages"] else ""

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(build_system_prompt()),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(user_prompt),
        ]
    )

    response = runnable.invoke(
        prompt_template.invoke(
            {
                "history": state["messages"][:-1],
                "input": build_prompt(user_prompt),
            }
        )
    )

    state["messages"].append(AIMessage(content=response.content))

    return state


# Function to evaluate the first interaction
def evaluate_interaction(state: State) -> str:
    if (
        "questions" not in state
        or not state["questions"]
        or len(state["messages"]) == 1
    ):
        return "agent"  # First interaction directs to introduction agent
    return "analyze_questions"  # Otherwise, analyze the user's input


# Main function to handle the chat
def handle_chat(user_prompt: str, form_id: str, thread_id: str):

    # Create the StateGraph and add nodes
    builder = StateGraph(state_schema=State)

    # Nodes
    builder.add_node("agent", agent)
    # Define the flow

    # Si no están inicializadas, agregar preguntas y luego ir a evaluate_interaction
    builder.add_edge(START, "agent")

    builder.add_edge("analyze_questions", "agent")

    # The END
    builder.add_edge("agent", END)

    # Compile and execute the LangGraph workflow
    app = builder.compile(checkpointer=memory)
    config = {
        "configurable": {
            "form_id": thread_id,
            "thread_id": thread_id,
        },
        memory: {},
    }

    final_state = app.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}, config
    )

    # Return the final AI response
    return final_state["messages"][-1].content
