from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
import logging
from git import Repo
import os
import shutil 
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad

GITHUB_REPO = "https://github.com/snap-stanford/UCE.git"
GIT_HASH = "7b31528b84e4c8e7a9717c61e3d03ff7559c61af"
BASE_DIR = Path(os.path.dirname(__file__)).parents[1]
MODEL_PATH = Path.joinpath(BASE_DIR, "data/33l_8ep_1024t_1280.torch")
ADATA_PATH = Path.joinpath(BASE_DIR, "data/full_cells_macaca.h5ad")
UCE_DST_PATH = Path.joinpath(BASE_DIR, "data/UCE")
ADATA_DST_PATH = Path.joinpath(BASE_DIR, "data/full_cells_macaca_uce_adata.h5ad")

class UCE(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")

        if UCE_DST_PATH.is_dir():
            self.log.info(f"Folder: {UCE_DST_PATH} exists already. Removing it...")
            shutil.rmtree(UCE_DST_PATH)

        self.log.info(f"Clonging UCE from GitHub: {GITHUB_REPO}")
        repo = Repo.clone_from(GITHUB_REPO, UCE_DST_PATH)
        repo.git.checkout(GIT_HASH)

    def run(self, species_name: str) -> None:
        '''
        Runs inference with the UCE model.
        
        Args:
            species_name: The name of the species.
        '''
        self.log.info(f"Inference started")

        # run from their folder
        os.chdir(UCE_DST_PATH)
        os.system(f"python3 eval_single_anndata.py --filter False --nlayers 33 --model_loc '{MODEL_PATH}' --adata_path '{ADATA_PATH}' --species '{species_name}'")
        output = Path.joinpath(BASE_DIR, "data/UCE/full_cells_macaca_uce_adata.h5ad")
        
        self.log.info(f"Inference ran successfully. Copying resulting {output} to {ADATA_DST_PATH}")
        shutil.copyfile(output, ADATA_DST_PATH)

    def get_embeddings(self) -> pd.DataFrame:
        '''
        Returns the embeddings after inference has been made.
        
        Returns:
            The embeddings in a pandas dataframe.
        '''
        try:
            ADATA_DST_PATH.resolve(strict=True)
        except FileNotFoundError:
            self.log.info(f"File not found error. Make sure {ADATA_DST_PATH} exists.")
            return np.empty
        else: 
            self.log.info(f"Loading {ADATA_DST_PATH} to get embeddings.")
            data = ad.read_h5ad(ADATA_DST_PATH)
            embeddings = data.obsm["X_uce"]
            return embeddings