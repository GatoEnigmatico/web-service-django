import os
import logging

from langchain.chat_models import ChatOpenAI
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg import openapi
from .models import Document, ChatSession, ChatMessage
from .serializers import DocumentSerializer
from .vector_service import search, db
from django.shortcuts import get_object_or_404
from django.http import JsonResponse

# Swagger documentation imports
from drf_yasg.utils import swagger_auto_schema

from langchain.schema import AIMessage, HumanMessage, SystemMessage
# Configure logger
logger = logging.getLogger(__name__)

class DocumentListCreateView(generics.ListCreateAPIView):
    """
    API endpoint for listing and creating documents.
    """
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer


class RAGSearchView(APIView):
    """
    API endpoint to perform RAG search using a query.
    """
    def get(self, request):
        query = request.GET.get("query", "")
        results = search(query)
        return Response({"results": results})


class DocumentCreateView(APIView):
    """
    API endpoint for creating a document.
    """
    def post(self, request):
        serializer = DocumentSerializer(data=request.data)
        if serializer.is_valid():
            document = serializer.save()

            # Generate embedding and store it in FAISS.
            # Alternatively, you can use: embeddings.embed_query(document.content)
            db.add_texts([document.content], metadatas=[{"title": document.title}])

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DocumentListView(APIView):
    """
    API endpoint for listing all documents.
    """
    def get(self, request):
        documents = Document.objects.all()
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)


class DocumentDeleteView(APIView):
    """
    API endpoint for deleting a document.
    """
    def delete(self, request, doc_id):
        try:
            document = Document.objects.get(id=doc_id)
            document.delete()

            # Also remove the document from the vector store.
            db.delete([document.content])

            return Response({"message": "Document deleted"}, status=status.HTTP_200_OK)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)


class DocumentRetrieveView(APIView):
    """
    API endpoint for retrieving a single document.
    """
    def get(self, request, doc_id):
        try:
            document = Document.objects.get(id=doc_id)
            serializer = DocumentSerializer(document)
            return Response(serializer.data)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)


class ChatHistoryView(APIView):
    """
    API endpoint for retrieving the chat history of a session.
    """
    def get(self, request, session_id):
        messages = ChatMessage.objects.filter(session__session_id=session_id).order_by("timestamp")
        chat_history = [{"role": msg.role, "content": msg.content} for msg in messages]
        return Response({"session_id": session_id, "history": chat_history})


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
        operation_description=(
            "This endpoint accepts a user prompt and returns an AI-generated response. "
            "It utilizes a FAISS vector store to retrieve relevant context for the prompt, "
            "and uses LangChain with OpenAI to generate the response. "
            "All prompt-related text (instructions given to the AI) are in Spanish."
        ),
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
        # Get the user prompt from the request data
        user_prompt = request.data.get('prompt', '')
        if not user_prompt:
            # If no prompt is provided, return a bad request response
            return Response({"error": "No prompt provided."}, status=400)

        # Retrieve conversation history from the session (if any)
        history = request.session.get('history', [])
        
        # Use FAISS search to retrieve relevant documents or text snippets for the prompt
        relevant_docs = search(user_prompt)
        # Combine the content of relevant documents (if any) into a single context string
        context_text = ""
        if relevant_docs:
            # `relevant_docs` might be a list of strings or LangChain Document objects
            context_parts = []
            for doc in relevant_docs:
                if isinstance(doc, str):
                    context_parts.append(doc)
                elif hasattr(doc, 'page_content'):
                    context_parts.append(doc.page_content)
                else:
                    # Fallback: use the string representation of the doc
                    context_parts.append(str(doc))
            context_text = "\n".join(context_parts)

        # Define the system instruction prompt in Spanish (prompt-related text)
        system_instructions = (
            "La siguiente es una conversación amistosa entre un humano y una IA. "
            "La IA es útil, habladora y proporciona muchos detalles específicos de su contexto. "
            "Si la IA no sabe la respuesta a una pregunta, dice honestamente que no la sabe."
        )
        # If there is relevant context from FAISS, include it in the system message
        if context_text:
            system_message_content = f"{system_instructions}\n\nInformación relevante proporcionada para ayudar a responder:\n{context_text}"
        else:
            system_message_content = system_instructions

        # Build the conversation as a list of messages for the chat model
        messages = [SystemMessage(content=system_message_content)]
        # Append previous conversation history to messages (if any)
        for entry in history:
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        # Append the latest user prompt as the last message
        messages.append(HumanMessage(content=user_prompt))

        # Initialize the OpenAI chat model via LangChain
        chat_model = ChatOpenAI(temperature=0.7)
        # Generate the AI response
        ai_response = chat_model(messages)
        answer_text = ai_response.content

        # Update the session history with the new question and answer
        history.append({"role": "user", "content": user_prompt})
        history.append({"role": "assistant", "content": answer_text})
        request.session['history'] = history  # Save updated history in session

        # Return the response as JSON
        return Response({"response": answer_text})