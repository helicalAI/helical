import logging
import pickle

import h5py
import numpy as np
import pandas as pd
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_embeddings(embeddings_path):
    with open(embeddings_path, "rb") as f:
        if embeddings_path.endswith(".pkl"):
            embeddings = pickle.load(f)
        elif embeddings_path.endswith(".h5"):
            embeddings = load_from_hdf5(embeddings_path)
    return embeddings


def load_from_hdf5(file_path):
    """Load dictionary from HDF5 file.

    Args:
        file_path (str): Path to HDF5 file containing embeddings data. The file should have
            a 'keys' dataset containing gene names and an 'arrays' group containing the
            corresponding embedding arrays.

    Returns
    -------
        dict: Dictionary mapping gene names (str) to their embedding arrays (numpy.ndarray).
            The keys are decoded from bytes to UTF-8 strings.
    """
    data_dict = {}
    with h5py.File(file_path, "r") as f:
        # Get the keys
        keys = [k.decode("utf-8") for k in f["keys"][:]]

        # Load the arrays
        arrays_group = f["arrays"]
        for key in keys:
            data_dict[key] = arrays_group[str(key)][:]

    return data_dict


def stack_dict(output):
    concatenated_data = {}
    for key in output[0].keys():
        if isinstance(output[0][key], torch.Tensor):
            if output[0][key].dim() == 0:  # Scalar tensor
                concatenated_data[key] = [batch[key].item() for batch in output]
            else:
                concatenated_data[key] = torch.cat(
                    [batch[key] for batch in output], dim=0
                )
        elif isinstance(output[0][key], np.ndarray):
            concatenated_data[key] = np.concatenate(
                [batch[key] for batch in output], axis=0
            )
        elif isinstance(output[0][key], dict):
            concatenated_data[key] = {
                k: np.vstack([batch[key][k] for batch in output]).flatten()
                for k in output[0][key].keys()
            }
        elif isinstance(output[0][key], int | float):  # Python scalar
            concatenated_data[key] = [batch[key] for batch in output]
        elif isinstance(output[0][key], list):
            concatenated_data[key] = sum([batch[key] for batch in output], [])
        else:  # Handle other types
            concatenated_data[key] = [batch[key] for batch in output]
    return concatenated_data


def save_as_hdf5(data_dict, output_path):
    """Save dictionary as HDF5 file."""
    with h5py.File(output_path, "w") as f:
        # Store the keys as a dataset
        keys = list(data_dict.keys())
        f.create_dataset("keys", data=np.array(keys, dtype="S"))

        # Create a group for the arrays
        arrays_group = f.create_group("arrays")
        for key, value in data_dict.items():
            arrays_group.create_dataset(str(key), data=value)


def filter_minimum_class(
    X: np.ndarray, y: np.ndarray | pd.Series, min_class_size: int = 10
) -> tuple[np.ndarray, np.ndarray | pd.Series]:
    logging.info(f"Label composition ({y.name}):")
    value_counts = y.value_counts()
    logging.info(f"Total classes before filtering: {len(value_counts)}")

    filtered_counts = value_counts[value_counts >= min_class_size]
    logging.info(
        f"Total classes after filtering (min_class_size={min_class_size}): {len(filtered_counts)}"
    )

    y = pd.Series(y) if isinstance(y, np.ndarray) else y
    class_counts = y.value_counts()

    valid_classes = class_counts[class_counts >= min_class_size].index
    valid_indices = y.isin(valid_classes)

    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]

    return X_filtered, pd.Categorical(y_filtered)
