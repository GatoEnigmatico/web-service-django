import json
import logging
import uuid  # Import to generate a unique thread_id

# Third-party imports
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

# Django imports
from django.shortcuts import get_object_or_404, render

# Local application imports
# from .models import Character
# from .serializers import CharacterSerializer
from .services.cifava_chat_service import handle_cifava_chat  # Import the chat logic

# Configure logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CharacterViewSet: CRUD endpoints for managing characters via DRF.
# class CharacterViewSet(viewsets.ModelViewSet):
#     """
#     API endpoint that allows characters to be viewed, created, updated, or deleted.
#     """

#     queryset = Character.objects.all()
#     serializer_class = CharacterSerializer

#     # Custom action to process a prompt for a specific character.
#     # Endpoint: /characters/<pk>/prompt/<conversation_id>/
#     @action(
#         detail=True, methods=["post"], url_path="prompt/(?P<conversation_id>[0-9]+)"
#     )
#     def prompt(self, request, pk=None, conversation_id=None):
#         # Retrieve the character object based on the primary key (pk)
#         # character = self.get_object()
#         prompt = request.data.get("prompt")
#         if not prompt:
#             return Response(
#                 {"error": "The 'prompt' field is required."},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#         # Call the GPT API using the character's configuration
#         response_text = ""
#         data = {
#             "character": "",
#             "conversation_id": conversation_id,
#             "prompt": prompt,
#             "response": response_text,
#         }
#         return Response(data)


# -----------------------------------------------------------------------------
# CharacterPromptAPIView: Endpoint to process a prompt for a character identified by its name and a conversation ID.
class CharacterPromptAPIView(APIView):
    """
    API endpoint to process a prompt for a character, identified by its unique name and a conversation ID.
    URL: /character/<name>/<conversation_id>/
    """

    def post(self, request, name, conversation_id, format=None):
        # Retrieve the character by its name and ensure it is active.
        # character = get_object_or_404(Character, name=name, active=True)

        # Extract the 'prompt' field from the request data.
        prompt = request.data.get("prompt")
        if not prompt:
            return Response(
                {"error": "The 'prompt' field is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Call the GPT API using the provided prompt and the character's configuration.
        response_text = ""

        # Return the response along with the character name and conversation ID.
        return Response(
            {
                "character": "",
                "conversation_id": conversation_id,
                "prompt": prompt,
                "response": response_text,
            }
        )


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Inicializar MemorySaver
memory = MemorySaver()

graph = create_react_agent(llm, tools=[], checkpointer=memory)


# Define a serializer to validate input
class ChatRequestSerializer(serializers.Serializer):
    prompt = serializers.CharField(required=True, allow_blank=False)
    system = serializers.CharField(required=False, allow_blank=True)

class ChatAPIView(APIView):
    """
    Legacy API endpoint for processing chat requests.
    """

    def post(self, request, format=None):
        # Validate input using serializer
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        # Log session details using logger instead of print
        for key, value in request.session.items():
            logger.debug(f"Session item: {key} => {value}")

        # Generate a thread_id if it does not exist in the session
        thread_id = request.session.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            request.session["thread_id"] = thread_id
            logger.debug(f"New thread_id generated: {thread_id}")

        config = {
            "configurable": {
                "thread_id": thread_id,
            },
        }

        # Build the messages list including the optional system message
        messages = []
        system_message = data.get("system", "")
        if system_message:
            # Retrieve state messages, defaulting to an empty list if key not present
            state_messages = graph.get_state(config).values.get("messages", [])
            
            if not isinstance(state_messages, list):
                raise ValueError("state_messages should be a list")

            if not len(state_messages):
                messages.append(SystemMessage(content=system_message))
            else:
                for i, item in enumerate(state_messages):
                    if isinstance(item, SystemMessage) and item.content != system_message:
                        graph.update_state(config,{"messages": [SystemMessage(content=system_message, id=item.id)]})

        # Append the human prompt message
        messages.append(HumanMessage(data["prompt"]))

        # Invoke the graph and handle potential exceptions
        try:
            final_state = graph.invoke({"messages": messages}, config)
            response_content = final_state["messages"][-1].content
        except Exception as e:
            logger.error("Error during graph invocation", exc_info=True)
            return Response(
                {"error": "An error occurred while processing your request."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Construct the response
        return Response(
            {
                "response": response_content,
            }
        )

class CIFAVAChatAPIView(APIView):
    """
    ChatView handles chat requests by retrieving context via a FAISS vector store
    and generating responses using LangChain's OpenAI integration.
    All prompt-related text (e.g., system instructions) is kept in Spanish,
    while code, comments, and documentation are in English.
    """

    # (If authentication is required, you can uncomment or modify the following lines)
    # authentication_classes = [JWTAuthentication]
    # permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        operation_summary="Chat with the AI assistant",
        operation_description="This endpoint accepts a user prompt and returns an AI-generated response.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "prompt": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="The user's prompt or question (in Spanish).",
                ),
            },
            required=["prompt"],
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "response": openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description="The AI's response to the prompt (in Spanish).",
                    )
                },
            )
        },
    )
    def post(self, request, format=None):
        # Get the user's message
        user_prompt = request.data.get("prompt", "")
        if not user_prompt:
            return Response(
                {"error": "No prompt provided."}, status=status.HTTP_400_BAD_REQUEST
            )
        for key, value in request.session.items():
            print("{} => {}".format(key, value))
        # Generate a `thread_id` if it does not exist in the session
        thread_id = request.session.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())  # Generate a unique UUID
            request.session["thread_id"] = thread_id  # Store it in the session

        # Call chat logic with `thread_id`
        ai_response = handle_cifava_chat(
            user_prompt, form_id=thread_id, thread_id=thread_id
        )

        # Return the AI's response
        return Response({"response": ai_response})
