from abc import ABC, abstractmethod
from helical.utils.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
from anndata import AnnData
import logging
from typing import List, Protocol, runtime_checkable
from datasets import Dataset
from numpy import ndarray
import torch
from helical.models.fine_tune.fine_tuning_heads import (
    ClassificationHead,
    RegressionHead,
    HelicalBaseFineTuningHead,
)
from typing import Literal
from typing import Union
from pandas import DataFrame

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class BaseModelProtocol(Protocol):
    def process_data(self, x: AnnData) -> Dataset: ...
    def get_embeddings(self, dataset: Dataset) -> ndarray: ...


class HelicalBaseFoundationModel(ABC, Logger):
    """Helical Base Foundation Model Class which serves as the base class for all foundation models in the helical package.
    Each new model should be a subclass of this class.

    Parameters
    ----------
    logging_type : LoggingType
        The logging type
    level : LoggingLevel
        The logging level

    Returns
    -------
    None

    """

    def __init__(self, logging_type=LoggingType.CONSOLE, level=LoggingLevel.INFO):

        super().__init__(logging_type, level)

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def get_embeddings():
        pass


class HelicalRNAModel(HelicalBaseFoundationModel):
    def ensure_rna_data_validity(
        self, adata: AnnData, gene_names: str, use_raw_counts: bool = True
    ) -> None:
        """Ensures that the data contains the gene_names and has integer counts for adata.X which is saved
        in 'total_counts'.

        Parameters
        ----------
        adata : AnnData
            The data to be checked.
        gene_names : str
            The name of the column containing gene names in adata.var.
        use_raw_counts : bool, default=True
            Whether to use raw counts or not.

        Raises
        ------
        KeyError
            If the data is missing column names.
        """

        if gene_names == "index":

            # as the gene_names is "index" by default, check that the data is strings
            if not all(isinstance(item, str) for item in adata.var.index):
                message = "The data in the index must only contain strings."
                LOGGER.error(message)
                raise ValueError(message)

            adata.var["index"] = adata.var.index

        # verify gene col name is present in adata.var
        if not gene_names in adata.var:
            message = f"Data must have the provided key '{gene_names}' in its 'var' section to be processed by the Helical RNA model."
            LOGGER.error(message)
            raise KeyError(message)

        # verify that the data in X are integers
        adata.obs["total_counts"] = adata.X.sum(axis=1)
        if use_raw_counts and not (adata.obs["total_counts"] % 1 == 0).all():
            message = "The data in X must be integers."
            LOGGER.error(message)
            raise ValueError(message)

    def get_valid_rna_sequence(
        self, sequences: Union[list[str], DataFrame]
    ) -> list[str]:
        """
        Returns valid RNA sequences that only contain the characters A, C, U, G, N and E.

        Parameters
        ----------
        sequences : List[str] or DataFrame
            The RNA sequences to be checked. If a DataFrame is provided, it should have a column named 'Sequence'.

        Raises
        ------
        ValueError
            If the sequences contain characters other than A, C, U, G, N and E.

        Returns
        -------
        List[str]
            Valid RNA sequences.
        """

        if isinstance(sequences, DataFrame):
            if "Sequence" not in sequences.columns:
                message = "The DataFrame must have a column named 'Sequence'."
                LOGGER.error(message)
                raise KeyError(message)
            sequences = sequences["Sequence"].tolist()

        valid_chars = set("ACUGNE")

        for sequence in sequences:
            if not set(sequence.upper()).issubset(valid_chars):
                invalid_chars = set(sequence.upper()) - valid_chars
                message = (
                    f"Invalid RNA sequence: found invalid characters {invalid_chars}"
                )
                LOGGER.error(message)
                raise ValueError(message)

        return sequences


class HelicalDNAModel(HelicalBaseFoundationModel):
    def get_valid_dna_sequence(
        self, sequences: Union[list[str], DataFrame], enforce_characters: bool = True
    ) -> list[str]:
        """
        Returns valid DNA sequences that only contain the characters A, C, T, G, N.

        Parameters
        ----------
        sequences : List[str]
            The DNA sequences to be checked.

        Raises
        ------
        ValueError
            If the sequences contain characters other than A, C, T, G, N and E.

        Returns
        -------
        List[str]
            Valid DNA sequences.
        """

        if isinstance(sequences, DataFrame):
            if "Sequence" not in sequences.columns:
                message = "The DataFrame must have a column named 'Sequence'."
                LOGGER.error(message)
                raise KeyError(message)

            sequences = sequences["Sequence"].tolist()

        if enforce_characters:
            valid_chars = set("ACTGN")

            for sequence in sequences:
                if not set(sequence.upper()).issubset(valid_chars):
                    invalid_chars = set(sequence.upper()) - valid_chars
                    message = f"Invalid DNA sequence: found invalid characters {invalid_chars}"
                    LOGGER.error(message)
                    raise ValueError(message)

        return sequences


class BaseTaskModel(ABC):
    """
    Helical Base Task Model which serves as the base class for all models trained for a specific task (such as classification).
    Each new model for a specific task should be a subclass of this class.
    """

    def __init__():
        pass

    @abstractmethod
    def compile(*args, **kwargs):
        pass

    @abstractmethod
    def train(*args, **kwargs):
        pass

    @abstractmethod
    def predict():
        pass

    @abstractmethod
    def save():
        pass

    @abstractmethod
    def load():
        pass


class HelicalBaseFineTuningModel(torch.nn.Module):
    """Helical Base Fine-Tuning Model Class which serves as the base class for all fine-tuning models in the helical package.
    Each new fine-tuning model should be a subclass of this class.

    Parameters
    ----------

    fine_tuning_head: Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model. This can either be a string (options available: "classification" and "regression") specifying the task or a custom fine-tuning head inheriting from HelicalBaseFineTuningHead.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    """

    def __init__(
        self,
        fine_tuning_head: (
            Literal["classification", "regression"] | HelicalBaseFineTuningHead
        ),
        output_size: int,
    ):
        super().__init__()
        if isinstance(fine_tuning_head, str):
            if fine_tuning_head == "classification":
                if output_size is None:
                    message = (
                        "The output_size must be specified for a classification head."
                    )
                    LOGGER.error(message)
                    raise ValueError(message)
                fine_tuning_head = ClassificationHead(output_size)
            elif fine_tuning_head == "regression":
                if output_size is None:
                    message = "The output_size must be specified for a regression head."
                    LOGGER.error(message)
                    raise ValueError(message)
                fine_tuning_head = RegressionHead(output_size)
            else:
                message = "Not implemented fine-tuning head."
                LOGGER.error(message)
                raise NotImplementedError(message)

        elif not isinstance(fine_tuning_head, HelicalBaseFineTuningHead):
            message = (
                "The fine_tuning_head must be a valid 'HelicalBaseFineTuningHead'."
            )
            LOGGER.error(message)
            raise ValueError(message)

        self.fine_tuning_head = fine_tuning_head

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def get_outputs():
        pass

    def save_model(self, path: str):
        """
        Save the model to a file.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        torch.save(self.model, path)
        LOGGER.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load the model from a file.

        Parameters
        ----------
        path : str
            The path to load the model from.
        """
        self.model = torch.load(path, weights_only=False)
        self.model.eval()
        self.fine_tuning_head.eval()
        LOGGER.info(f"Model loaded from {path}")
