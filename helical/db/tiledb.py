import tiledbsoma.io
import tiledbsoma
import logging 
from helical.services.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
import shutil
from pathlib import Path

class TileDB(Logger):
    
    def __init__(self, loging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(loging_type, level)
        self.log = logging.getLogger("TileDB")
    
    def generate_tiledb_soma(self, input_path: Path, tiledb_folder_name: str, measurement_name: str) -> str:
        '''
        Generates a tiledb soma database

        Args:
            input_uri: Path to the h5ad file.
            tiledb_uri: The URI to where the database will be.
            measurement_name: The name of the measurement.
        
        Returns:
            A URI to the created tiledb
        '''
        if os.path.isdir(tiledb_folder_name):
            self.log.info(f"TileDB folder: {tiledb_folder_name} exists already. Removing it...")
            shutil.rmtree(tiledb_folder_name)

        uri = tiledbsoma.io.from_h5ad(experiment_uri = tiledb_folder_name, input_path = input_path, measurement_name = measurement_name)
        self.log.info(f"Successfully created a TileDB in: {tiledb_folder_name}")
        return uri
