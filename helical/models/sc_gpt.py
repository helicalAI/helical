from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
from pathlib import Path
from helical.services.downloader import Downloader

BASE_DIR = Path(os.path.dirname(__file__)).parents[1]

class SCGPT(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("scGPT-Model")
        self.downloader = Downloader()

        # paths and file names
        self.scgpt_dst_path = Path.joinpath(BASE_DIR, "data/sc-GPT")

    def get_model(self):
        self.downloader.clone_git_repo(self.scgpt_dst_path,
                                       "https://github.com/bowang-lab/scGPT.git",
                                       "v0.2.1")

    def run(self):
        pass

    def get_embeddings(self):
        pass