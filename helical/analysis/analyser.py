from helical.db.tiledb import TileDB
from pathlib import Path
from helical.services.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import tiledbsoma
import numpy as np


class Analyser(Logger):
    def __init__(self,
                 h5ad_input: Path,
                 tiledb_folder_name: str,
                 measurement_name: str,
                 loging_type = LoggingType.CONSOLE, 
                 level = LoggingLevel.INFO) -> None:
        super().__init__(loging_type, level)
        self.log = logging.getLogger("Analyser")
        
        # TODO: First try opening. If that fails, generate. This makes it easier as it will be one-time only
        # Pfizer
        # uri = TileDB().generate_tiledb_soma(input_path=h5ad_input,
        #                             tiledb_folder_name = tiledb_folder_name,
        #                             measurement_name = measurement_name)
        uri = "./data/macaca"
        self.experiment_pf = tiledbsoma.open(uri)

        # UCE
        # uri = TileDB().generate_tiledb_soma(input_path="./data/IMA_sample.h5ad",
        #                             tiledb_folder_name = "./data/ima_sample",
        #                             measurement_name = "RNA")
        uri = "./data/ima_sample"
        self.experiment_uce = tiledbsoma.open(uri)

        for o in self.experiment_uce.obs.read([slice(0, 1000000)], value_filter = "(species == 'human' and (tissue == 'blood' or tissue == 'thalamic complex')) or species == 'macaca_fascicularis'",):
            self.df = o.to_pandas()


    def generate_sample(self):
        obs = self.experiment_pf.obs
        for o in obs.read([self.df['soma_joinid'].values.tolist()]):
            labels_exp = o.to_pandas()
        
        X = self.experiment_pf['ms']['RNA']['obsm']['X_uce']
        n_obs = len(self.experiment_pf['obs'])
        n_var = 1280
        tensor = X.read().coos((n_obs, n_var)).concat().to_scipy()

        tmp = tensor.todense()
        embeddings_pf = np.squeeze(np.asarray(tmp))
        print(embeddings_pf.shape)
