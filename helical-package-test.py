from helical.models.uce import UCE
from helical.models.sc_gpt import SCGPT
from helical.preprocessor import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.transform_table(input_path='/Users/bputzeys/Documents/Helical/ETS_data/21iT009_051_full_data.csv', 
                                 output_path='./data/full_cells_macaca.h5ad',
                                 mapping_path='./data/ensemble_to_display_name_batch_macaca.pkl',
                                 count_column='rcnt')
    
    uce = UCE()
    
    uce.get_model()
    uce.run("macaca_fascicularis")
    embeddings = uce.get_embeddings()

    print(embeddings.shape)

    # WIP but general idea: Have different models at disposition to run inference
    # scgpt = SCGPT()
    
    # scgpt.get_model()
    # scgpt.run("macaca_fascicularis")
    # embeddings = scgpt.get_embeddings()

    # print(embeddings.shape)

