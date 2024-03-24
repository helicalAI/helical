from fastapi import APIRouter, File, UploadFile
import pandas as pd
import json
import io
from fastapi.responses import Response
from pathlib import Path
from ..run import Run
from fastapi.responses import StreamingResponse
router = APIRouter()

run = Run()

@router.post("/post_gene_expression_csv")
def post_gene_expressions(file: UploadFile = File(...)) -> Response:
    df = pd.read_csv(file.file)
    run.init_data(df)
    file.file.close()
    return {"Gene expression CSV file successfully loaded!"}

@router.post("/post_model_config")
def post_model_config(file: UploadFile = File(...)) -> Response:
    json_data = json.load(file.file)
    run.init_model(json_data["model_config"], json_data["data_config"], json_data["files_config"])
    return {"data_in_file": json_data}

@router.get("/get_embeddings")
def get_embeddings() -> Response:
    embeddings = run.run_uce()
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df = embeddings_df.reset_index()
    stream = io.StringIO()
    embeddings_df.to_csv(stream, index = False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                )
    response.headers["Content-Disposition"] = "attachment; filename=embeddings.csv"
    return response 