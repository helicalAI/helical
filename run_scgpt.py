from helical.models.scgpt.scgpt_helical import SCGPT

import json

with open('./scgpt_config.json') as f:
    config = json.load(f)

scgpt = SCGPT(config["model_config"],
          config["data_config"])

data_loader = scgpt.process_data()
embeddings = scgpt.get_embeddings()

print(embeddings.shape)