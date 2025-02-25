import torch
from evo2 import Evo2

model = Evo2("evo2_1b_base")
# model = torch.load(
#     "/home/matthew/.cache/huggingface/evo2-1b-base.pt", map_location="cpu"
# )

# Then save it with the new zipfile serialization option
# torch.save(
#     model,
#     "/home/matthew/.cache/huggingface/evo2-1b-base.pt",
#     _use_new_zipfile_serialization=True,
# )
