from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.views.generic import TemplateView
from .views import CIFAVAChatAPIView, ChatAPIView


from django.views.generic import TemplateView
from .views import CharacterPromptAPIView

# Configure the DRF router for the Character endpoints.
router = DefaultRouter()
# router.register(r"characters", CharacterViewSet)

urlpatterns = [
    # CRUD endpoints for characters (list, create, retrieve, update, delete)
    path("", include(router.urls)),
    # Custom endpoint to process a prompt based on the character's name and conversation ID.
    # URL: /character/<name>/<conversation_id>/
    path(
        "character/<str:name>/<int:conversation_id>/",
        CharacterPromptAPIView.as_view(),
        name="character-prompt",
    ),
    # LEGACY endpoint: maintained for backward compatibility
    path("chat/", ChatAPIView.as_view(), name="chat"),
    # IAV endpoints: example static page and chat endpoint
    path("iav/cifava/", CIFAVAChatAPIView.as_view(), name="chat"),
    path(
        "iav/cifava", TemplateView.as_view(template_name="static_page.html"), name="iav"
    ),
]


# urlpatterns = [
    # path('documents/', DocumentListCreateView.as_view(), name='document-list-create'),
    # path('search/', RAGSearchView.as_view(), name='rag-search'),
    # path('documents/', DocumentListView.as_view(), name='document-list'),
    # path('documents/add/', DocumentCreateView.as_view(), name='document-create'),
    # path('documents/<int:doc_id>/', DocumentRetrieveView.as_view(), name='document-detail'),
    # path('documents/delete/<int:doc_id>/', DocumentDeleteView.as_view(), name='document-delete'),
# ]
