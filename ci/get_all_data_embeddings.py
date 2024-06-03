import numpy as np
from datasets import DatasetDict
from tqdm import tqdm
from helical.models.hyena_dna.model import HyenaDNA
from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig    
from datasets import get_dataset_config_names
from datasets import load_dataset
import os

configurer = HyenaDNAConfig(model_name="hyenadna-tiny-1k-seqlen-d256")
hyena_model = HyenaDNA(configurer=configurer)


def get_model_inputs(dataset: DatasetDict, ratio: float = 1.0):
    
    x = np.empty((0, configurer.config['d_model'])) 
    labels = np.empty((0,), dtype=int)

    # disable logging to avoid cluttering the output
    import logging
    logging.disable(logging.CRITICAL)

    # use tqdm for a progress bar
    length = int(len(dataset)*ratio)
    for i in tqdm(range(length)):
        sequence = dataset["sequence"][i]
        
        tokenized_sequence = hyena_model.process_data(sequence)
        embeddings = hyena_model.get_embeddings(tokenized_sequence)
        
        numpy_array = embeddings[0].detach().numpy()
        mean_array = numpy_array.mean(axis=0)
        x = np.append(x, [mean_array], axis=0)
        
    # normalize the data
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    labels = np.array(dataset["label"][:length])
    return x, labels

labels = get_dataset_config_names("InstaDeepAI/nucleotide_transformer_downstream_tasks")

for label in labels:
    dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", label)

    x, y = get_model_inputs(dataset["train"], 0.001)
    
    if not os.path.exists("data"):
        os.makedirs("data")

    np.save(f"data/x_{label}_norm_256", x)
    np.save(f"data/y_{label}_norm_256", y)

    X_unseen, y_unseen = get_model_inputs(dataset["test"], 0.001)
    np.save(f"data/x_unseen_{label}_norm_256", X_unseen)
    np.save(f"data/y_unseen_{label}_norm_256", y_unseen)
    break


