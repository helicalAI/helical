from helical.benchmark.benchmark import Benchmark
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
from helical.models.uce.model import UCE
from helical.classification.neural_network import NeuralNetwork
from helical.classification.svm import SupportVectorMachine as SVM
from helical.classification.classifier import Classifier
import anndata as ad
from omegaconf import DictConfig
import hydra
import json
import requests
import os

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:
    geneformer = Geneformer()
    scgpt = scGPT()
    uce = UCE()

    train_data = ad.read_h5ad("./examples/notebooks/c_data.h5ad")
    eval_data = ad.read_h5ad("./examples/notebooks/ms_default.h5ad")

    train_data = train_data[:10]
    eval_data = eval_data[:10]

    train_data.var["ensembl_id"] = train_data.var["index_column"]
    train_data.obs['n_counts'] = train_data.X.sum(axis=1)

    eval_data.var["ensembl_id"] = train_data.var["index_column"]
    eval_data.obs['n_counts'] = train_data.X.sum(axis=1)

    geneformer_c = Classifier().train_classifier_head(train_data,
                                                      geneformer, 
                                                      NeuralNetwork(**cfg["neural_network"]), 
                                                      gene_col_name="ensembl_id",
                                                      labels_column_name="celltype",
                                                      test_size=0.1,
                                                      random_state=42)   
    geneformer_c.trained_task_model.save("cell_type_annotation/geneformer/geneformer_model.h5")
    geneformer_c.trained_task_model.save_encoder("cell_type_annotation/geneformer/geneformer_encoder")

    scgpt_c = Classifier().train_classifier_head(train_data, 
                                                 scgpt, 
                                                 NeuralNetwork(**cfg["neural_network"]),
                                                 gene_col_name="index",
                                                 labels_column_name="celltype",
                                                 test_size=0.1,
                                                 random_state=42)       
    scgpt_c.trained_task_model.save("cell_type_annotation/scgpt/scgpt_model.h5")
    scgpt_c.trained_task_model.save_encoder("cell_type_annotation/scgpt/scgpt_encoder")     

    uce_c = Classifier().train_classifier_head(train_data, 
                                               uce, 
                                               NeuralNetwork(**cfg["neural_network"]),
                                               gene_col_name="index",
                                               labels_column_name="celltype",
                                               test_size=0.1,
                                               random_state=42)
    uce_c.trained_task_model.save("cell_type_annotation/uce/uce_model.h5")
    uce_c.trained_task_model.save_encoder("cell_type_annotation/uce/uce_encoder")   

    bench = Benchmark()
    evaluations = bench.evaluate_classification([geneformer_c], eval_data, "celltype")
    
    # Serializing json
    json_object = json.dumps(evaluations, indent=4)
    
    # Writing to sample.json
    with open("cell_type_annotation/cell_type_annotation.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    def download_files(files: list[str])-> None:

        for filename in files:
            url = f"https://helicalpackage.blob.core.windows.net/helicalpackage/data/{filename}"

            # Check if the file already exists in the current directory
            if os.path.exists(filename):
                print(f"Files already exist. Skipping downloads.")
            else:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filename, "wb") as file:
                        file.write(response.content)
                    print(f"Downloaded {filename} successfully.")
                else:
                    print(f"Failed to download {filename}.")
    files = ["c_data.h5ad", "ms_default.h5ad"]
    download_files(files)
    benchmark()