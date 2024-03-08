from rest_framework.decorators import api_view
from rest_framework.response import Response
from helical.models.uce import UCE
from pathlib import Path

@api_view(['GET'])
def get_embeddings(request):
    embeddings = UCE().get_embeddings(Path("<absolut-path-to-your-annotated-h5ad-data>"))
    return Response({'message': f'The embeddings are {embeddings}'})