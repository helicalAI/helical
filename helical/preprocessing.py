import requests
import json
import pickle as pkl
import pandas as pd

def save_ensemble_to_display_name(path_to_ets_csv, output):
    '''
    Saves a mapping of the `Ensemble ID` to `display names`. 
    '''

    df = pd.read_csv(path_to_ets_csv)
    genes = df['egid'].dropna().unique()

    server = "https://rest.ensembl.org"
    ext = "/lookup/id"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}

    ensemble_to_display_name = dict()

    for i in range(0, len(genes), 1000):
        print(i+1000,"/",len(genes))
        ids = {'ids':genes[i:2].tolist()}
        r = requests.post(server+ext, headers=headers, data=json.dumps(ids))
        decoded = r.json()
        ensemble_to_display_name.update(decoded)

    pkl.dump(ensemble_to_display_name, open(output, 'wb')) 

if __name__ == "__main__":
    save_ensemble_to_display_name('./21iT009_051_full_data.csv', './ensemble_to_display_name_batch_macaca.pkl')
    

