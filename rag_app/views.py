import os
import logging

from langchain.llms import OpenAI
from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response

from .models import Document, ChatSession, ChatMessage
from .serializers import DocumentSerializer
from .vector_service import search, db

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
    API endpoint for handling chat interactions.
    """
    def post(self, request):
        session_id = request.data.get("session_id")
        query = request.data.get("query", "")

        if not query:
            return Response({"error": "You must provide a question."}, status=status.HTTP_400_BAD_REQUEST)

        # Create or retrieve the chat session
        session, _ = ChatSession.objects.get_or_create(session_id=session_id)

        # Retrieve chat history
        previous_messages = ChatMessage.objects.filter(session=session).order_by("timestamp")
        history = [{"role": msg.role, "content": msg.content} for msg in previous_messages]

        # 🔍 Search for relevant documents using FAISS (RAG)
        retrieved_docs = search(query)
        if retrieved_docs:
            context_docs = "\n\n".join([f"- {doc['content']}" for doc in retrieved_docs])
        else:
            context_docs = "No relevant documents found."

        # Log retrieved documents for debugging
        logger.debug(f"Retrieved documents: {context_docs}")

        # Build the messages list for OpenAI's ChatCompletion API.
        # Note: the prompt sent to ChatGPT remains in Spanish.
        messages = [
            # {
            #     "role": "system",
            #     "content": (
            #         "Eres un asistente inteligente que responde preguntas utilizando la información relevante proporcionada. "
            #         "Proporciona respuestas claras y útiles basadas en el contexto."
            #     )
            # },
            # {
            #     "role": "system",
            #     "content": f"Documentos relevantes:\n{context_docs}"
            # }
        ]

        # Include previous chat history if available (in Spanish)
        if history:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            messages.append({
                "role": "system",
                "content": f"Historial de chat:\n{history_text}"
            })

        # Add the user's new question
        messages.append({"role": "user", "content": query})

        # 🔥 Call OpenAI's ChatCompletion API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,  # Adjust temperature as needed
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return Response({"error": "Error generating response"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save the conversation to the database
        ChatMessage.objects.create(session=session, role="user", content=query)
        ChatMessage.objects.create(session=session, role="assistant", content=response_text)

        return Response({
            "session_id": session.session_id,
            "query": query,
            "response": response_text
        })