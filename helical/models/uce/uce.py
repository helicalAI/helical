import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader

from helical.services.downloader import Downloader
from helical.preprocessor import Preprocessor
from helical.models.uce.uce_model import TransformerModel
from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings

class UCE(HelicalBaseModel):
    
    def __init__(self, model_config, data_config, files_config, accelerator=None, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")
        self.downloader = Downloader()
        self.preprocessor = Preprocessor()
        self.model_config = model_config
        self.data_config = data_config
        self.files_config = files_config

        self.embeddings = get_ESM2_embeddings(self.files_config)
        self.model =  load_model(self.model_config, self.embeddings)
        self.model = self.model.eval()
        self.accelerator = accelerator
        if accelerator is not None:
           self.model = accelerator.prepare(self.model)


    def get_model(self) -> TransformerModel:
        '''
        Returns:
            The model
        '''

        return self.model

    def process_data(self, data: AnnData, species="macaca_fascicularis") -> DataLoader:
        '''
        Process the data for the UCE model.

        Args:
            pfizer_csv: The path to the pfizer csv file.

        Returns:
            A path to the processed data.
        '''
        loader = process_data(data, model_config=self.model_config, species=species, accelerator=self.accelerator)
        return loader

    def run(self, dataloader: DataLoader) -> np.array:
        '''
        Runs inference with the UCE model.
        
        Args:
            model_path: The path to the UCE model.
            adata_input: The path to the processed adata h5ad file.
            species_name: The name of the species.
        
        Returns:
            A path to the h5ad data having the embeddings if inference ran successfully. 
            None otherwise.
        '''
        self.log.info(f"Inference started")

        embeddings = get_gene_embeddings(self.model, dataloader, self.accelerator)
        return embeddings

    def get_embeddings(self, dataloader:DataLoader) -> np.array:
        '''
        Returns the embeddings after inference has been made.
        
        Args:
            inferred_data_path: Path to the file the model outputted

        Returns:
            The embeddings in a pandas dataframe.
        '''
        embeddings = self.run(dataloader)
        return embeddings

    # def get_embeddings(self, inferred_data_path: Path) -> pd.DataFrame:
    #     '''
    #     Returns the embeddings after inference has been made.
        
    #     Args:
    #         inferred_data_path: Path to the file the model outputted

    #     Returns:
    #         The embeddings in a pandas dataframe.
    #     '''
    #     try:
    #         inferred_data_path.resolve(strict=True)
    #     except FileNotFoundError:
    #         self.log.info(f"File not found error. Make sure {inferred_data_path} exists.")
    #         return np.empty
    #     else: 
    #         self.log.info(f"Loading {inferred_data_path} to get embeddings.")
    #         data = ad.read_h5ad(inferred_data_path)
    #         embeddings = data.obsm["X_uce"]
    #         return embeddings