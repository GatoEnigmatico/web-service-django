from django.urls import path
from .views import (
    ChatView,
)
from .rag import search_rag
from django.views.generic import TemplateView

urlpatterns = [
    # path('documents/', DocumentListCreateView.as_view(), name='document-list-create'),
    # path('search/', RAGSearchView.as_view(), name='rag-search'),
    # path('documents/', DocumentListView.as_view(), name='document-list'),
    # path('documents/add/', DocumentCreateView.as_view(), name='document-create'),
    # path('documents/<int:doc_id>/', DocumentRetrieveView.as_view(), name='document-detail'),
    # path('documents/delete/<int:doc_id>/', DocumentDeleteView.as_view(), name='document-delete'),
    path("chat/", ChatView.as_view(), name="chat"),
    path("iav/cifava", TemplateView.as_view(template_name="static_page.html"), name="iav"),
    # path('chat/history/<str:session_id>/', ChatHistoryView.as_view(), name='chat-history'),
]
