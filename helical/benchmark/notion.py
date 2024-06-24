import os
from notion_client import Client
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")

# Initialize the client
notion = Client(auth=NOTION_TOKEN)

task_map = {
      "cell_type_annotation": "Cell Type Annotation",
      "batch_effect_correction": "Batch Effect Correction",
}

task_file_names = ["cell_type_annotation", "batch_effect_correction"]
for task_file_name in task_file_names:

    with open(f'helical/benchmark/{task_file_name}.json', 'r') as openfile:
        evaluations = json.load(openfile)
     
    page_id_to_model = {}
    for model in evaluations.keys():
        model_result = notion.search(query=model).get("results")
        id = model_result[0]["id"]
        page_id_to_model.update({id: model})

    task_result = notion.search(query=task_map[task_file_name]).get("results")
    assert len(task_result) == 1, f"Expected 1 task for {task_map[task_file_name]} but found {len(task_result)} tasks."
    id_task = task_result[0]["id"]

    for metric in ["Accuracy", "Precision", "F1", "Recall"]:
        score_results = notion.search(query=metric).get("results")
        for result in score_results:
            models_per_score = result["properties"]["Models"]["relation"]
            for model in models_per_score:
                if result["properties"]["Tasks"]["relation"][0]["id"] == id_task:
                    model_score_task = page_id_to_model[models_per_score[0]["id"]]
                    id=result["id"]
                    notion.pages.update(
                        id,
                        properties={
                            "Score": {
                                "number": evaluations[model_score_task][metric]
                            },
                            "Date": {
                                "date": {
                                    "start": datetime.now().isoformat(),
                                },
                            },
                        },
                    )
        
        
