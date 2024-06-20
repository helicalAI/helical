## Instructions
We use these files in this folder to try and replicate the results of the [Hyena](https://arxiv.org/pdf/2306.15794) and the [Nucleotide transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1.full.pdf) papers.
As there are 18 datasets, it takes some time to download the datasets, get the embeddings and train a neural network that makes predictions. This is the reason why we split the task in 2:

## 1. Download the dataset and get the embeddings
We use HuggingFace to access the datasets of the [nucleotide transformer downstream tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks). For each dataset, we pass the input sequence of nucleotides though the HyenaDNA model and save the output as a `.npy` file. This is done with the [get_all_data_embeddings.py](get_all_data_embeddings.py) which we call with the GitHub Actions file [get_embeddings.yml](../../.github/workflows/get_embeddings.yml) which is in the `.github/workflows` folder. The resulting `data` folder can be downloaded as an artifact.

## 2. Make predictions
With the embeddings for each dataset, we can train a neural network using the [predict_all.py](predict_all.py) file. As described in the [Hyena paper](https://arxiv.org/pdf/2306.15794), "the Matthews correlation coefficient (MCC) is used as the performance metric for the enhancer and epigenetic
marks dataset, and the F1-score is used for the promoter and splice site dataset". 
