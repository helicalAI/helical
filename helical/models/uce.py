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

BASE_DIR = Path(os.path.dirname(__file__)).parents[1]

class UCE(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")
        self.downloader = Downloader()

        # paths and file names
        self.uce_dst_path = Path.joinpath(BASE_DIR, "data/UCE")
        self.model_path = Path.joinpath(BASE_DIR, "data/33l_8ep_1024t_1280.torch")
        self.adata_input_path = Path.joinpath(BASE_DIR, "data/full_cells_macaca.h5ad")
        self.adata_dst_path = Path.joinpath(BASE_DIR, "data/full_cells_macaca_uce_adata.h5ad")

    def get_model(self) -> None:
        '''
        Gets the necessary ensemble mappings, clones the GitHub repo of the UCE and downloads the Pytorch UCE model itself.
        '''
        
        self.downloader.get_ensemble_mapping(Path.joinpath(BASE_DIR, 'data/21iT009_051_full_data.csv'), 
                                             Path.joinpath(BASE_DIR, 'data/ensemble_to_display_name_batch_macaca.pkl'))
        self.downloader.clone_git_repo(self.uce_dst_path, "https://github.com/snap-stanford/UCE.git", "7b31528b84e4c8e7a9717c61e3d03ff7559c61af")
        self.downloader.download_via_link(self.model_path, "https://figshare.com/ndownloader/files/43423236")

    def run(self, species_name: str) -> None:
        '''
        Runs inference with the UCE model.
        
        Args:
            species_name: The name of the species.
        '''
        self.log.info(f"Inference started")

        # run from their folder
        os.chdir(self.uce_dst_path)
        os.system(f"python3 eval_single_anndata.py --filter False --nlayers 33 --model_loc '{self.model_path}' --adata_path '{self.adata_input_path}' --species '{species_name}'")
        output = Path.joinpath(BASE_DIR, "data/UCE/full_cells_macaca_uce_adata.h5ad")
        
        self.log.info(f"Inference ran successfully. Copying resulting {output} to {self.adata_dst_path}")
        shutil.copyfile(output, self.adata_dst_path)

    def get_embeddings(self) -> pd.DataFrame:
        '''
        Returns the embeddings after inference has been made.
        
        Returns:
            The embeddings in a pandas dataframe.
        '''
        try:
            self.adata_dst_path.resolve(strict=True)
        except FileNotFoundError:
            self.log.info(f"File not found error. Make sure {self.adata_dst_path} exists.")
            return np.empty
        else: 
            self.log.info(f"Loading {self.adata_dst_path} to get embeddings.")
            data = ad.read_h5ad(self.adata_dst_path)
            embeddings = data.obsm["X_uce"]
            return embeddings