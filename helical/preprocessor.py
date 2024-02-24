import requests
import json
import pickle as pkl
import pandas as pd
import sys
print(sys.path)
# from services.logger import Logger
# from constants.enums import LoggingType, LoggingLevel
# import logging

# INTERVAL = 1000

# class Preprocessor(Logger):
#     def __init__(self, loging_type = LoggingType.NOTSET, level = LoggingLevel.NOTSET) -> None:
#         super().__init__(loging_type, level)
#         self.log = logging.getLogger("Preprocessor")

#     def save_ensemble_mapping(self, path_to_ets_csv: str, output: str) -> bool:
#         '''
#         Saves a mapping of the `Ensemble ID` to `display names`. 
        
#         Args:
#             path_to_ets_csv: Path to the ETS csv file.
#             output: Path to where the output (.pkl) file should be saved to.

#         Returns:
#             bool if successfull
#         '''
#         try:
#             df = pd.read_csv(path_to_ets_csv)
#         except:
#             self.log.exception(f"Failed to open the {path_to_ets_csv} file. Please provide it.")
#             return False
       
#         genes = df['egid'].dropna().unique()

#         server = "https://rest.ensembl.org/lookup/id"
#         headers={ "Content-Type" : "application/json", "Accept" : "application/json"}

#         ensemble_to_display_name = dict()
        
#         self.log.info(f"Starting to get the mappings of {len(genes)} genes from {server}.")

#         for i in range(0, len(genes), INTERVAL):
#             self.log.info(str(i+INTERVAL) + "/" + str(len(genes)))
#             ids = {'ids':genes[i:i+INTERVAL].tolist()}
#             r = requests.post(server, headers=headers, data=json.dumps(ids))
#             decoded = r.json()
#             ensemble_to_display_name.update(decoded)

#         self.log.info(f"Got all mappings. Saving to {output}.")
#         pkl.dump(ensemble_to_display_name, open(output, 'wb')) 
#         return True

# if __name__ == "__main__":
#     preprocessor = Preprocessor(LoggingType.CONSOLE, LoggingLevel.INFO)
#     preprocessor.save_ensemble_mapping('./21iT009_051_full_data.csv', './ensemble_to_display_name_batch_macaca.pkl')
    

