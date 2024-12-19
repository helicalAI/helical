# Helical Examples

The `examples` folder contains a `run_models` and a `notebooks` folder.

## Run Models
We show that each supported Helical model can be included in a uniform manner.
```
from helical.models.scgpt.model import scGPT, scGPTConfig

scgpt_config = scGPTConfig(batch_size=10)
scgpt = scGPT(configurer = scgpt_config)
```
For specific configurations, such as `batch_size`, a model can be provided with its own configuration (`scGPTConfig` in this case).
Processing the data and getting the embeddings is uniform across models too:
```
hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
ann_data = get_anndata_from_hf_dataset(hf_dataset)
data = scgpt.process_data(ann_data)
embeddings = scgpt.get_embeddings(data)
```

## Notebooks
To do something useful with these embeddings, we provide a number of use case examples in the `notebooks` folder.

One such example is the [Cell-Type-Annotation](./notebooks/Cell-Type-Annotation.ipynb) notebook. An scGPT model is used to get embeddings of a gene expression profile which are then used as inputs to a smaller neural network, predicting the cell type.

That notebook explains the procedure step-by-step in much detail.