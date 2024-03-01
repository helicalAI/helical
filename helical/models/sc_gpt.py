from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
import logging
from git import Repo
import os
import shutil 
from pathlib import Path

GITHUB_REPO = "https://github.com/bowang-lab/scGPT.git"
TAG = "v0.2.1"
BASE_DIR = Path(os.path.dirname(__file__)).parents[1]
SCGPT_DST_PATH = Path.joinpath(BASE_DIR, "data/SC-GPT")
ADATA_DST_PATH = Path.joinpath(BASE_DIR, "data/full_cells_macaca_uce_adata.h5ad")

class SCGPT(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")

        if SCGPT_DST_PATH.is_dir():
            self.log.info(f"Folder: {SCGPT_DST_PATH} exists already. Removing it...")
            shutil.rmtree(SCGPT_DST_PATH)

        self.log.info(f"Clonging SC-GPT from GitHub: {GITHUB_REPO}")
        repo = Repo.clone_from(GITHUB_REPO, SCGPT_DST_PATH)
        repo.git.checkout(TAG)
    
    def run(self):
        pass
