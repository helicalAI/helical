from helical.preprocessor import Preprocessor
from helical.downloader import Downloader
from helical.constants.enums import LoggingType

if __name__ == "__main__":
    # downloader = Downloader(loging_type=LoggingType.FILE)
    # downloader.get_ensemble_mapping('./data/21iT009_051_full_data.csv', './data/ensemble_to_display_name_batch_macaca.pkl')
    # downloader.download_via_link("./data/test", "https://figshare.com/ndownloader/files/42706555")
    preprocessor = Preprocessor(loging_type=LoggingType.CONSOLE)
    # preprocessor.transform_table(input_path='/Users/bputzeys/Documents/Helical/ETS_data/21iT009_051_full_data.csv', 
    #                              output_path='./data/full_cells_macaca.h5ad',
    #                              mapping_path='./data/ensemble_to_display_name_batch_macaca.pkl',
    #                              count_column='rcnt')
    preprocessor.generate_tiledb_soma(input_path='./data/full_cells_macaca.h5ad',
                                      tiledb_folder_name='./data/macaca',
                                      measurement_name='RNA')
    