from fastapi import APIRouter
from fastapi.responses import Response
from helical.models.uce import UCE
from pathlib import Path

router = APIRouter()

@router.get("/get_embeddings")
def get_embeddings() -> Response:
    embeddings = UCE().get_embeddings(Path("<absolut-path-to-your-annotated-h5ad-data>"))
    return Response(f'The embeddings are {embeddings}')