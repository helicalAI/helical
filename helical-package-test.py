from helical.preprocessor import Preprocessor
from helical.services.downloader import Downloader
from helical.constants.enums import LoggingType
from helical.db.tiledb import TileDB
from pathlib import Path
from helical.models.uce import UCE

if __name__ == "__main__":
    downloader = Downloader()
    downloader.get_ensemble_mapping('./data/21iT009_051_full_data.csv', './data/ensemble_to_display_name_batch_macaca.pkl')
    downloader.download_via_link("./data/33l_8ep_1024t_1280.torch", "https://figshare.com/ndownloader/files/43423236")

    preprocessor = Preprocessor()
    preprocessor.transform_table(input_path='/Users/bputzeys/Documents/Helical/ETS_data/21iT009_051_full_data.csv', 
                                 output_path='./data/full_cells_macaca.h5ad',
                                 mapping_path='./data/ensemble_to_display_name_batch_macaca.pkl',
                                 count_column='rcnt')

    UCE().run("macaca_fascicularis")

    TileDB().generate_tiledb_soma(input_path=Path('./data/full_cells_macaca_uce_adata.h5ad'),
                                  tiledb_folder_name='./data/macaca',
                                  measurement_name='RNA')