# Model Card for scGPT

## Model Details

**Model Name:** scGPT 

**Model Version:** 1.0 

**Model Description:** scGPT is a large-scale self-supervised transformer-based model, pre-trained across more than 33 million human cells under non-disease conditions. It is designed to perform various tasks, including cell type annotation, multi-batch integration, multi-omic integration, in silico perturbation response prediction, and gene regulatory network inference. The model is pre-trained on extensive single-cell RNA sequencing data to build a foundational understanding of cellular biology.

## Model Developers

**Developed By:** Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan, Bo Wang. See specific [author contributions](#citation) 

**Contact Information:** Bo Wang (bowang@vectorinstitute.ai) 

**License:** MIT License Copyright (c) 2022 suber 

## Model Type

**Architecture:** Transformer-based  

**Domain:** Cell Biology, Bioinformatics  

**Languages:** Single-cell transcriptomics data 


## Model Purpose

**Technical usage:**

- Tokenizing transcriptomes
- Tokenizing conditions (i.e. meta-information associated with individual genes, like perturbation experiment alterations, which are indicated by perturbation tokens)
- Pre-training
- Fine-tuning 
- Extracting and plotting cell embeddings

**Broader research applications:**  

- Cell type annotation 
- Perturbation response prediction
- Batch correction on integrating multiple scRNA-seq datasets
- Integrative representation learning for single-cell multi-omic data
- Gene regulatory network inference 

## Training Data

**Data Sources:** 

- Publicly available datasets are described in [data availability](https://www.nature.com/articles/s41592-024-02201-0#data-availability) in the manuscript

**Data Volume:**  

- Pre-trained on data from over 33 million human cells under non-disease conditions. This comprehensive dataset encompasses a wide range of cell types from 51 organs or tissues, and 441 studies

**Preprocessing:**  

- Normalization and scaling to ensure consistency across datasets
- Value binning technique to convert all expression counts into relative values

## Model Performance

**Evaluation Metrics:**  

- Classification metrics: Accuracy, Precision, Recall, Macro F1 
- Biological conservation metrics: NMIcell, ARIcell, ASWcell
- Batch correction metrics: ASWbatch, GraphConn

**Testing Data:**  

- Held-out subsets of the training dataset
- Additional external validation datasets from independent studies

## Model Limitations

**Known Limitations:**

- The current pretraining does not inherently mitigate batch effects, and thus the model’s zero-shot performance could be constrained on datasets with substantial technical variation
- Evaluating the model is also complex, given the frequent absence of definitive biological ground truths and the variation in data quality

**Future Improvements:**  

- Pretraining on a larger-scale dataset with more diversity, including multi-omic data, spatial omics and various diseased conditions
- Incorporation of perturbation and temporal data in the pretraining stage, enabling the model to learn causal relationships and infer how genes and cells respond to changes over time
- Development of techniques that allow the pretrained model to understand and adapt to different tasks and contexts in a zero-shot setting without the need for fine-tuning

## How to Use

**Input Format:**  

- The input to scGPT consists of three main components: (1) gene (or peak) tokens, (2) expression values (cell-by-gene matrix) and (3) condition tokens

**Output Format:**  

- Gene and cell embeddings, JSON format with predicted cell types and integrated multi-modal data

**Example Usage:**
```python
from helical.models.scgpt import scGPT, scGPTConfig
import anndata as ad

scgpt_config = scGPTConfig(batch_size=10)
scgpt = scGPT(configurer = scgpt_config)
adata = ad.read_h5ad("dataset.h5ad")
data = scgpt.process_data(adata)
embeddings = scgpt.get_embeddings(data)

print(embeddings.shape)
```

**Example Fine-Tuning:**

```python
from helical.models.scgpt import scGPTFineTuningModel, scGPTConfig

# Load the desired dataset
adata = ad.read_h5ad("dataset.h5ad")

# Get the desired label class
cell_types = list(ann_data.obs.cell_type)

# Get unique labels
label_set = set(cell_types)

# Create the fine-tuning model with the relevant configs
scgpt_config=scGPTConfig(batch_size=10)
scgpt_fine_tune = scGPTFineTuningModel(scGPT_config=scgpt_config, fine_tuning_head="classification", output_size=len(label_set))

# Process the data for training
data = scgpt_fine_tune.process_data(adata)

# Create a dictionary mapping the classes to unique integers for training
class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

for i in range(len(cell_types)):
    cell_types[i] = class_id_dict[cell_types[i]]

# Fine-tune
scgpt_fine_tune.train(train_input_data=dataset, train_labels=cell_types)
```

## Developers

Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan, Bo Wang

## Contact

- Bo Wang (bowang@vectorinstitute.ai)

## Citation

To cite the scGPT model, please use the following reference:
```bibtex
@article{cui2023scGPT,
  title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

For more details and updates, visit the [scGPT GitHub repository](https://github.com/bowang-lab/scGPT).

## Author contributions

H.C. developed the concept of the work and contributed to design and implementation of the algorithm. C.W. and K.P. contributed to design and implementation of the algorithm. H C., C.W., H.M., K.P. and F.L. contributed to the analysis of computational experiments. H.C. and C.W. drafted the initial version of the manuscript. H.C., C.W., H.M., K.P., F.L. and B.W. contributed to revision of the work. N.D. contributed to design of the algorithm. B.W. contributed to the conception and design of the work.