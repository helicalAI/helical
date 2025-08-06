# Model Card for Geneformer

## Model Details

**Model Name:** Geneformer  
**Model Versions:** 1.0 and 2.0  
**Model Description:** Geneformer is a context-aware, attention-based deep learning model pretrained on a large-scale corpus of single-cell transcriptomes. It is designed to enable context-specific predictions in settings with limited data in network biology. The model performs various downstream tasks such as gene network mapping, disease modeling, and therapeutic target identification.

In version 2.0, Geneformer introduces a cancer-tuned model variant using domain-specific continual learning. This variant was developed to address the exclusion of malignant cells from the initial pretraining due to their propensity for gain-of-function mutations. The cancer-tuned model underwent additional training with ~14 million cells from cancer studies, including matched healthy controls, to provide contrasting context. This approach allows the model to better understand gene network rewiring in malignancy while maintaining its general knowledge of gene network dynamics.

When to use each model:
- Base pretrained model: Use for general transcriptomic analysis tasks and non-cancer-specific applications.
- Cancer-tuned model: Use for cancer-specific analyses, tumor microenvironment studies, and predicting factors that could shift cells to tumor-restricting or immune-activating states.

## Model Versions

Geneformer has two main versions:

**Version 1.0:**

- Pretrained on approximately 30 million single-cell transcriptomes
- Input size of 2048 genes per cell
- Focused on single-task learning

**Version 2.0:**

- Pretrained on Genecorpus-103M, comprising ~103 million human single-cell transcriptomes
- Initial self-supervised pretraining with ~95 million cells, excluding cells with high mutational burdens
- Expanded input/context size of 4096 genes per cell
- Employs multi-task learning to jointly learn cell types, tissues, disease states, and developmental stages
- Includes a cancer-tuned model variant using domain-specific continual learning
- Supports model quantization for resource-efficient fine-tuning and inference

Key improvements in v2.0:
- Larger and more diverse pretraining corpus
- Increased model parameters and expanded input size
- Multi-task learning for context-specific representations of gene network dynamics
- Improved zero-shot predictions in diverse downstream tasks
- Cancer-specific tuning for tumor microenvironment analysis

## Available Models for each Version

### Version 1.0 (30M dataset)
- **gf-6L-10M-i2048**
    - 6 layers
    - 2048 input size
    - Trained on ~30 million cells
- **gf-12L-40M-i2048**
    - 12 layers
    - 2048 input size
    - Trained on ~30 million cells
- **gf-12L-40M-i2048-CZI-CellxGene**
    - 12 layers
    - 2048 input size
    - Pretrained on ~30 million cells
    - Fine-tuned by the CELLxGENE Discover Team at CZI using multi-task learning on the CELLxGENE Census
    - Optimized to distinguish context-specific cell states across cell types, tissues, developmental stages, and diseases
    - [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/)
    - [CELLxGENE Discover Team](mailto:soma@chanzuckerberg.com)

### Version 2.0 (95M dataset)
- **gf-12L-38M-i4096**
    - 12 layers
    - 4096 input size
    - Trained on ~95 million cells
- **gf-20L-151M-i4096**
    - 20 layers
    - 4096 input size
    - Trained on ~95 million cells
- **gf-12L-38M-i4096-CLcancer**
    - 12 layers
    - 4096 input size
    - Initially trained on ~95 million cells
    - Further tuned on ~14 million cancer cells
- **gf-12L-104M-i4096**
    - 12 layers
    - 4096 input size
    - Initially trained on ~104M human single cell transcriptomes
- **gf-12L-104M-i4096-CLcancer**
    - 12 layers
    - 4096 input size
    - Initially trained on ~104M human single cell transcriptomes
    - Further tuned on ~14 million cancer cells
- **gf-18L-316M-i4096**
    - 18 layers
    - 4096 input size
    - Initially trained on ~104M human single cell transcriptomes

## Model Developers

**Developed by:** 
Christina V. Theodoris conceived of the work, developed Geneformer, assembled Genecorpus-30M and designed and performed computational analyses.  

**Contact Information:** 
christina.theodoris@gladstone.ucsf.edu  

**License:** 
Apache-2.0 

## Model Type

**Architecture:** 
Transformer-based  

**Domain:** 
Cell Biology, Bioinformatics  

**Input Data:** 
Single-cell transcriptomes

## Model Purpose

**Technical usage:**

- Tokenizing transcriptomes
- Pre-training
- Hyperparameter tuning
- Fine-tuning
- Extracting and plotting cell embeddings
- In silico perturbation

**Broader research applications:**  

- Research in genomics and network biology
- Disease modeling with limited patient data
- Identification of candidate therapeutic targets
- Prediction of gene dosage sensitivity and chromatin dynamics
- Context-specific predictions in gene regulatory networks

## Training Data

**Data Sources:**  

- Publicly available single-cell transcriptomic datasets
- GeneCorpus-30M is available on the [Hugging Face Dataset Hub](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M)
- Genecorpus-103M will be available on Hugging Face Dataset Hub (coming soon)

**Data Volume:**  

- Version 1.0: 29.9 million single-cell transcriptomes across a wide range of tissues
- Version 2.0: ~103 million human single-cell transcriptomes (Genecorpus-103M), including:
    - ~95 million cells for initial self-supervised pretraining
    - ~14 million cells from cancer studies for domain-specific continual learning

**Preprocessing:**  

- Exclusion of cells with high mutational burdens
- Metrics established for scalable filtering to exclude possible doublets and/or damaged cells
- Rank value encoding of transcriptomes where genes are ranked by scaled expression within each cell

## Model Performance

**Evaluation Metrics:**  

- Predictive accuracy in distinguishing:
    - With *fine-tuning*:
        - Transcription factor dosage sensitivity
        - Chromatin dynamics (bivalently marked promoters)
        - Transcription factor regulatory range
        - Gene network centrality
        - Transcription factor targets
        - Cell type annotation
        - Batch integration
        - Cell state classification across differentiation
        - Disease classification
        - In silico perturbation to determine disease-driving genes
        - In silico treatment to determine candidate therapeutic targets
    - With *Zero-shot learning*:
        - Batch integration
        - Gene context specificity
        - In silico reprogramming
        - In silico differentiation
        - In silico perturbation to determine impact on cell state
        - In silico perturbation to determine transcription factor targets
        - In silico perturbation to determine transcription factor cooperativity

**Testing Data:**  

- Held-out subsets of the training dataset
- Additional validation using publicly available datasets 
- Experimental validation for: 
    - Prediction of novel transcription factor in cardiomyocytes with zero-shot learning that had a functional impact on cardiomyocytes' contractile force generation 
    - Prediction of candidate therapeutic targets with in silico treatment analysis that significantly improved contractile force generation of cardiac microtissues in an iPS cell model of cardiomyopathy

## Model Limitations

**Known Limitations:**  

- May not generalize well to newly discovered tissues or rare gene variants
- Performance may vary across different single-cell sequencing technologies

**Future Improvements:**  

- Integration of new data sources to improve model robustness
- Enhancements in model architecture to better handle diverse transcriptomic profiles

## How to Use

**Input Format:**  

- Rank value encoded single-cell transcriptomes

**Output Format:**  

- Contextual gene and cell embeddings
- Contextual attention weights
- Contextual predictions

**Example Usage:**
```python
from helical.models.geneformer import Geneformer, GeneformerConfig
import anndata as ad

# Example configuration
model_config = GeneformerConfig(model_name="gf-12L-38M-i4096", batch_size=10)
geneformer_v2 = Geneformer(model_config)

# Example usage for base pretrained model
ann_data = ad.read_h5ad("anndata_file.h5ad")
dataset = geneformer_v2.process_data(ann_data)
embeddings = geneformer_v2.get_embeddings(dataset)
print("Base model embeddings shape:", embeddings.shape)

# Example usage for cancer-tuned model
model_config_cancer = GeneformerConfig(model_name="gf-12L-38M-i4096-CLcancer", batch_size=10)
geneformer_v2_cancer = Geneformer(model_config)

cancer_ann_data = ad.read_h5ad("anndata_file.h5ad")
cancer_dataset = geneformer_v2_cancer.process_data(cancer_ann_data)
cancer_embeddings = geneformer_v2_cancer.get_embeddings(cancer_dataset)
print("Cancer-tuned model embeddings shape:", cancer_embeddings.shape)
```

**Example Fine-Tuning:**

```python
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
import anndata as ad

# Load the data
ann_data = ad.read_h5ad("/home/matthew/helical-dev/helical/yolksac_human.h5ad")

# Get the column for fine-tuning
cell_types = list(ann_data.obs["cell_types"])
label_set = set(cell_types)

# Create a GeneformerConfig object
geneformer_config = GeneformerConfig(model_name="gf-12L-38M-i4096", batch_size=10)

# Create a GeneformerFineTuningModel object
geneformer_fine_tune = GeneformerFineTuningModel(geneformer_config=geneformer_config, fine_tuning_head="classification", output_size=len(label_set))

# Process the data
dataset = geneformer_fine_tune.process_data(ann_data[:10])

# Add column to the dataset
dataset = dataset.add_column('cell_types', cell_types)

# Create a dictionary to map cell types to ids
class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

def classes_to_ids(example):
    example["cell_types"] = class_id_dict[example["cell_types"]]
    return example

# Convert cell types to ids
dataset = dataset.map(classes_to_ids, num_proc=1)

# Fine-tune the model
geneformer_fine_tune.train(train_dataset=dataset, label="cell_types")

# Get logits from the fine-tuned model
outputs = geneformer_fine_tune.get_outputs(dataset)
print(outputs[:10])

# Get embeddings from the fine-tuned model
embeddings = geneformer_fine_tune.get_embeddings(dataset)
print(embeddings[:10])
```

## Contact

christina.theodoris@gladstone.ucsf.edu

## Citation

```bibtex
@article{theodoris2023transfer,
  title={Transfer learning enables predictions in network biology},
  author={Theodoris, Christos V and Xiao, Liang and Chopra, Arun and Chaffin, Mark D and Al Sayed, Zainab R and Hill, Michael C and Mantineo, Hannah and Brydon, Ellen M and Zeng, Zexian and Liu, X Shirley and Ellinor, Patrick T},
  journal={Nature},
  volume={618},
  pages={616--624},
  year={2023},
  doi={10.1038/s41586-023-06139-9}
}
```

```bibtex
@article{chen2024quantized,
  title={Quantized multi-task learning for context-specific representations of gene network dynamics},
  author={Chen, H and Venkatesh, M S and Gomez Ortega, J and Mahesh, S V and Nandi, T and Madduri, R and Pelka, K and Theodoris, C V},
  journal={bioRxiv},
  year={2024},
  month={aug},
  day={19},
  doi={10.1101/2024.08.16.608180},
  note={co-first authors: H Chen, M S Venkatesh; co-senior authors: K Pelka, C V Theodoris; corresponding author: C V Theodoris}
}
```
## Author contributions 


C.V.T. conceived of the work, developed Geneformer, assembled Genecorpus-30M and designed and performed computational analyses. L.X., A.C., Z.R.A.S., M.C.H., H.M. and E.M.B. performed experimental validation in engineered cardiac microtissues. M.D.C. performed preprocessing, cell annotation and differential expression analysis of the cardiomyopathy dataset. Z.Z. provided data from the TISCH database for inclusion in Genecorpus-30M. X.S.L. and P.T.E. designed analyses and supervised the work. C.V.T., X.S.L. and P.T.E. 

HC and MSV developed the models and designed/performed computational analyses. HC assembled the pretraining corpuses and developed the continual learning method. MSV developed the multi-task learning method and quantization strategy. JGO contributed to model pretraining and corpus assembly. 
