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

analyce_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "Eres un asistente que revisa respuestas de usuarios a preguntas predefinidas."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template(
            'La pregunta a evaluar es: "{question}". '
            "Revisa la conversación y responde con 'sí' si la pregunta ha sido respondida, "
            "de lo contrario, responde 'no'."
        ),
    ]
)

# trimmer = trim_messages(
#     max_tokens=45,
#     strategy="last",
#     token_counter=llm,
#     # Usually, we want to keep the SystemMessage
#     # if it's present in the original history.
#     # The SystemMessage has special instructions for the model.
#     include_system=True,
#     # Most chat models expect that chat history starts with either:
#     # (1) a HumanMessage or
#     # (2) a SystemMessage followed by a HumanMessage
#     # start_on="human" makes sure we produce a valid chat history
#     start_on="human",
# )

runnable = llm

# bound_model = runnable.bind_tools(tools)


# Función para finalizar el flujo
def always_end(state: State) -> str:
    return END


def add_questions_node(state: State) -> State:
    """Agrega preguntas si aún no están en el estado."""

    # Solo agregar preguntas si el estado no las tiene aún
    if "questions" not in state or not state["questions"]:
        state["questions"] = QUESTIONS

    return state


def get_next_unanswered_question(state: State) -> Question:
    """Obtiene la siguiente pregunta sin responder, o devuelve None si todas han sido contestadas."""
    for question in state["questions"]:
        if question["answer"] is None:
            return question
    return None  # No hay preguntas pendientes


def agent(state: State) -> State:

    if "questions" not in state or not state["questions"]:
        state = add_questions_node(state)

    user_prompt = state["messages"][-1].content if state["messages"] else ""

    next_question = get_next_unanswered_question(state)

    if next_question == None:
        return state

    response = runnable.invoke(
        prompt_template.invoke(
            {
                "history": state["messages"][:-1],
                "input": build_prompt(
                    user_prompt=user_prompt,
                    question=next_question["question"],
                ),
            }
        )
    )

    state["messages"].append(AIMessage(content=response.content))

    return state


def analyze_questions(state: State) -> State:
    """Analiza si el usuario respondió la pregunta actual y actualiza el state."""

    # Obtener la última respuesta del usuario
    user_message = state["messages"][-1].content if state["messages"] else ""

    # Obtener la pregunta sin contestar más reciente
    next_question = get_next_unanswered_question(state)

    if not next_question:
        return state  # No hay preguntas pendientes, no hacemos nada

    # Construimos el prompt para el modelo de OpenAI
    prompt = analyce_prompt_template.invoke(
        {
            "question": next_question["question"],
            "history": state["messages"],
            "input": user_message,
        }
    )

    analysis_response = runnable.invoke(prompt)
    answered_correctly = analysis_response.content.lower().strip() == "sí"

    # Si el modelo dice que la respuesta es válida, asignamos la respuesta al estado
    if answered_correctly:
        for question in state["questions"]:
            if question["key"] == next_question["key"]:
                question["answer"] = user_message  # Guardamos la respuesta
                break

    return state  # Devolvemos el estado actualizado


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
    builder.add_node("add_questions", add_questions_node)
    builder.add_node("agent", agent)
    builder.add_node("analyze_questions", analyze_questions)
    builder.add_node("always_end", always_end)
    # Define the flow

    # Si no están inicializadas, agregar preguntas y luego ir a evaluate_interaction
    builder.add_conditional_edges(
        START, evaluate_interaction, ["agent", "analyze_questions"]
    )

    builder.add_edge("analyze_questions", "agent")

    # The END
    builder.add_edge("agent", END)

    # Compile and execute the LangGraph workflow
    app = builder.compile(checkpointer=memory)
    config = {
        "configurable": {
            "form_id": form_id,
            "thread_id": 8,
        },
        memory: {},
    }

    final_state = app.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}, config
    )

    # Return the final AI response
    return final_state["messages"][-1].content
