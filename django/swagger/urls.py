from django.urls import path
from .views import get_embeddings

urlpatterns = [
    path('get-embeddings/', get_embeddings, name='get-embeddings/'),
]