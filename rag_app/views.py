import logging
import uuid  # Import to generate a unique thread_id
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg import openapi
from .services.chat_service import handle_chat  # Import the chat logic

# Swagger documentation imports
from drf_yasg.utils import swagger_auto_schema

# Configure logger
logger = logging.getLogger(__name__)


class ChatView(APIView):
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
                'prompt': openapi.Schema(
                    type=openapi.TYPE_STRING, 
                    description="The user's prompt or question (in Spanish)."
                ),
            },
            required=['prompt']
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'response': openapi.Schema(
                        type=openapi.TYPE_STRING, 
                        description="The AI's response to the prompt (in Spanish)."
                    )
                }
            )
        }
    )
    def post(self, request, format=None):
        # Get the user's message
        user_prompt = request.data.get('prompt', '')
        if not user_prompt:
            return Response({"error": "No prompt provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve conversation history
        history = request.session.get('history', [])

        # Generate a `thread_id` if it does not exist in the session
        thread_id = request.session.get('thread_id')
        if not thread_id:
            thread_id = str(uuid.uuid4())  # Generate a unique UUID
            request.session['thread_id'] = thread_id  # Store it in the session

        # Call chat logic with `thread_id`
        ai_response, updated_history = handle_chat(user_prompt, history, thread_id=thread_id)

        # Update the session history
        request.session['history'] = updated_history

        # Return the AI's response
        return Response({"response": ai_response})