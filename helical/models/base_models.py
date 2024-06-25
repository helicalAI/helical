from abc import ABC, abstractmethod
from helical.services.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
from anndata import AnnData
import logging
from typing import Protocol, runtime_checkable
from datasets import Dataset
from numpy import ndarray

LOGGER = logging.getLogger(__name__)

@runtime_checkable
class BaseModelProtocol(Protocol):
    def process_data(self, x: AnnData) -> Dataset:
        ...
    def get_embeddings(self, dataset: Dataset) -> ndarray:
        ...
        
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
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO):

        super().__init__(logging_type, level)

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def get_embeddings():
        pass

class HelicalRNAModel(HelicalBaseFoundationModel):
    def check_rna_data_validity(self, adata: AnnData, gene_col_name: str) -> None:
        """Checks if the data is contains the gene_col_name, which is needed for all Helical RNA models.  

        Parameters
        ----------
        adata : AnnData
            The data to be checked.
        gene_col_name : str
            The name of the column containing gene names in adata.var.

        Raises
        ------
        KeyError
            If the data is missing column names.
        """
        
        if gene_col_name == "index":
    
            # as the gene_col_name is "index" by default, check that the data is strings
            if not all(isinstance(item, str) for item in adata.var.index):
                message = "The data in the index must only contain strings."
                LOGGER.error(message)
                raise ValueError(message)
    
            adata.var["index"] = adata.var.index
        
        # verify gene col name is present in adata.var
        if not gene_col_name in adata.var:
            message = f"Data must have the provided key '{gene_col_name}' in its 'var' section to be processed by the Helical RNA model."
            LOGGER.error(message)
            raise KeyError(message)
        
class HelicalDNAModel(HelicalBaseFoundationModel):
    def check_dna_data_validity(self) -> None:
        pass # TODO

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