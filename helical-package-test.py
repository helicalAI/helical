from helical.models.uce.uce import UCE
from helical.models.sc_gpt import SCGPT
from pathlib import Path
    
uce = UCE()

model_path = uce.get_model()
processed_data = uce.process_data(Path('<your/absolute/path/goes/here>'))
result = uce.run(model_path, processed_data, "macaca_fascicularis")
embeddings = uce.get_embeddings(result)

print(embeddings.shape)

# WIP but general idea: Have different models at disposition to run inference
# scgpt = SCGPT()

# scgpt.get_model()
# scgpt.run("macaca_fascicularis")
# embeddings = scgpt.get_embeddings()

# print(embeddings.shape)

