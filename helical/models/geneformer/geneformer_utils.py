import anndata as ad
import pickle as pkl
import requests, sys
import json
import pickle as pkl




def load_mappings(gene_symbols):
    server = "https://rest.ensembl.org"
    ext = "/lookup/symbol/homo_sapiens"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}

    # r = requests.post(server+ext, headers=headers, data='{ "symbols" : ["BRCA2", "BRAF" ] }')

    gene_id_to_ensemble = dict()

    for i in range(0, len(gene_symbols), 1000):
        # print(i+1000,"/",len(test.var['gene_symbols']))
        symbols = {'symbols':gene_symbols[i:i+1000].tolist()}
        r = requests.post(server+ext, headers=headers, data=json.dumps(symbols))
        decoded = r.json()
        gene_id_to_ensemble.update(decoded)
        # print(repr(decoded))

    pkl.dump(gene_id_to_ensemble, open('./human_gene_to_ensemble_id.pkl', 'wb'))
    return gene_id_to_ensemble