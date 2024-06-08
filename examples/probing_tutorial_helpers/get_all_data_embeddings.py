import numpy as np
from datasets import DatasetDict
from helical.models.hyena_dna.model import HyenaDNA
from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig    
from datasets import get_dataset_config_names
from datasets import load_dataset
import os
from typing import Tuple
from tqdm import tqdm

configurer = HyenaDNAConfig(model_name="hyenadna-tiny-1k-seqlen-d256")
hyena_model = HyenaDNA(configurer=configurer)

def get_model_inputs(dataset: DatasetDict, nbr_sequences: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    dataset : DatasetDict
        The dataset containing the sequences and labels.
    nbr_sequences : int, optional
        The number of sequences to process at a time, by default 50.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the input features and labels.
    """
    x_tot = np.empty((0, configurer.config['d_model'])) 
    labels_tot = np.empty((0,), dtype=int)
    
    # disable logging to avoid cluttering the output
    import logging
    logging.disable(logging.CRITICAL)

    for i in tqdm(range(0, len(dataset), nbr_sequences)):
        tokenized_sequences = hyena_model.process_data(dataset["sequence"][i:i+nbr_sequences])    
        embeddings = hyena_model.get_embeddings(tokenized_sequences[0])
        numpy_array = embeddings.cpu().detach().numpy()
        x = numpy_array.mean(axis=1)
            
        # normalize the data
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        x_tot = np.concatenate((x_tot, x), axis=0)

        labels = np.array(dataset["label"][i:i+nbr_sequences])
        labels_tot = np.concatenate((labels_tot, labels), axis=0)
    return x_tot, labels_tot

labels = get_dataset_config_names("InstaDeepAI/nucleotide_transformer_downstream_tasks")

for i, label in enumerate(labels):
    print(f"Processing '{label}' dataset: {i+1} of {len(labels)}")

    dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks", label)
    x, y = get_model_inputs(dataset["train"], 50)

    if not os.path.exists("data"):
        os.makedirs("data/train")
        os.makedirs("data/test")


    np.save(f"data/train/x_{label}_norm_256", x)
    np.save(f"data/train/y_{label}_norm_256", y)

    X_unseen, y_unseen = get_model_inputs(dataset["test"], 50)
    np.save(f"data/test/x_{label}_norm_256", X_unseen)
    np.save(f"data/test/y_{label}_norm_256", y_unseen)
