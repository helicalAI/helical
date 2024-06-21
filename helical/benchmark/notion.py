import os
import sys
from notion_client import Client
from dotenv import load_dotenv
import json

load_dotenv()
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")

# Initialize the client
notion = Client(auth=NOTION_TOKEN)

with open('helical/benchmark/results.json', 'r') as openfile:
	evaluations = json.load(openfile)
     
model_to_page_id = {}
for model in evaluations.keys():
     model_result = notion.search(query=model).get("results")
     id = model_result[0]["id"]
     model_to_page_id.update({id: model})

score_results = notion.search(query="Accuracy").get("results")

for result in score_results:
    id=result["id"]
    notion.pages.update(
        id,
        properties={
            "Score": {
                 "number": evaluations[model_to_page_id[result["properties"]["Models"]["relation"][0]["id"]]]["Accuracy"]
            }
        },
    )
