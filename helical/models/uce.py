from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
import shutil 
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from helical.services.downloader import Downloader
from helical.preprocessor import Preprocessor

BASE_DIR = Path(os.path.dirname(__file__)).parents[1]

class UCE(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")
        self.downloader = Downloader()
        self.preprocessor = Preprocessor()

        # paths and file names
        self.uce_dst_path = Path.joinpath(BASE_DIR, "data/UCE")
        self.model_path = Path.joinpath(BASE_DIR, "data/33l_8ep_1024t_1280.torch")
        self.adata_dst_path = Path.joinpath(BASE_DIR, "data/full_cells_macaca_uce_adata.h5ad")
        self.mapping_file = Path.joinpath(BASE_DIR, 'data/ensemble_to_display_name_batch_macaca.pkl')

    def get_model(self) -> Path:
        '''
        Clones the GitHub repo of the UCE and downloads the Pytorch UCE model.

        Returns:
            A path to where the model is saved.
        '''

        self.downloader.clone_git_repo(self.uce_dst_path, "https://github.com/snap-stanford/UCE.git", "7b31528b84e4c8e7a9717c61e3d03ff7559c61af")
        self.downloader.download_via_link(self.model_path, "https://figshare.com/ndownloader/files/43423236")
        return self.model_path

    def process_data(self, pfizer_csv: Path) -> Path:
        '''
        Process the data for the UCE model.

        Args:
            pfizer_csv: The path to the pfizer csv file.

        Returns:
            A path to the processed data.
        '''
        self.downloader.get_ensemble_mapping(pfizer_csv, self.mapping_file)
        processed = self.preprocessor.transform_table(pfizer_csv,
                                                      self.mapping_file,
                                                      'rcnt')
        return processed

    def run(self, model_path: Path, adata_input: Path, species_name: str) -> Path:
        '''
        Runs inference with the UCE model.
        
        Args:
            model_path: The path to the UCE model.
            adata_input: The path to the processed adata h5ad file.
            species_name: The name of the species.
        
        Returns:
            A path to the h5ad data having the embeddings 
        '''
        self.log.info(f"Inference started")

        # run from their folder
        os.chdir(self.uce_dst_path)
        os.system(f"python3 eval_single_anndata.py --filter False --nlayers 33 --model_loc '{model_path}' --adata_path '{adata_input}' --species '{species_name}'")
        output = Path.joinpath(BASE_DIR, "data/UCE/full_cells_macaca_uce_adata.h5ad")
        
        self.log.info(f"Inference ran successfully. Copying resulting {output} to {self.adata_dst_path}")
        shutil.copyfile(output, self.adata_dst_path)
        return self.adata_dst_path

    def get_embeddings(self, inferred_data_path: Path) -> pd.DataFrame:
        '''
        Returns the embeddings after inference has been made.
        
        Args:
            inferred_data_path: Path to the file the model outputted

        Returns:
            The embeddings in a pandas dataframe.
        '''
        try:
            inferred_data_path.resolve(strict=True)
        except FileNotFoundError:
            self.log.info(f"File not found error. Make sure {inferred_data_path} exists.")
            return np.empty
        else: 
            self.log.info(f"Loading {inferred_data_path} to get embeddings.")
            data = ad.read_h5ad(inferred_data_path)
            embeddings = data.obsm["X_uce"]
            return embeddings