# Model Card for STATE

## Model Details

**Model Name:** STATE

**Model Version:** 0.9.14

**Model Description:** 

## Model Developers

**Developed By:** Adduri et al. see [preprint](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1) for complete author list and contributions.


**Contact Information:** Yusuf Roohani (yusuf.roohani@arcinstitute.org) 

**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)

## Model Type

**Architecture:** Transformer-based  

**Domain:** Cell Biology, Bioinformatics  

**Languages:** Single-cell transcriptomics data 

## Model Purpose

**Technical usage:**

- Embedding transcriptomes
- Embedding conditions (i.e. perturbation experiment alterations, which are indicated by perturbation tokens)
- Pre- and post-perturbation single-cell embeddings

**Broader research applications:**  

- Perturbation response prediction
- Embedding single-cell expression data for downstream tasks 

## Training Data

**Data Sources:**

- Publicly available datasets and code are described in the preprint's code and model availability section.

**Data Volume:**

- Pre-trained on data from 167 million human cells for the embedding model and over 100 million cells for the perturbation model. The perturbed cells are chemically or genetically perturbed cells from large scale single-cell screens.

**Preprocessing:**  

- Normalization and scaling to ensure consistency across datasets

## Model Performance

**Evaluation Metrics:**  

- Classification metrics: Accuracy, Precision, Recall
- Biological conservation metrics: Cell-Eval package (incl. p-values, fold change, ranking, correlation)
- Batch correction metrics: Batch Correction Embeddings

**Testing Data:**  

- Held-out subsets of the training dataset

## Model Limitations

**Known Limitations:**

- Single-cell RNA sequencing data requires the destruction of cells during measurement. This prevents observations of their non-perturbed states. Some perturbations such as gene knockout may also not occur experimentally and be incorrectly flagged as a pertubed datapoint. 

- STATE evaluation metrics are more sensitive on an individual gene basis but stronger on an ensemble/batch level. As pointed out by the authors metrics such as accuracy of individual DE genes depends on dataset size/quality.

- Attention maps are sensitive to cell set heterogeneity.

**Future Improvements:**  

- Improving perturbation featurisation between modalities such as drugs and gene knockdown, which would allow for combinatorial perturbation predictions.

- STATE has been trained on over 70 cell contexts but not tested for cell context not completely seen during training (e.g., unseen cell types).

- Training of the STATE embedding model on larger datasets.

## How to Use

**Input Format:**  

- Annotated data matrix (.anndata)

**Output Format:**  

- Cell embedding/transcriptome (.anndata, .npy)

**Example State Embedding Usage:**
```python

from helical.models.state import StateEmbed
from helical.models.state import StateConfig
import scanpy as sc

state_config = StateConfig(batch_size=16)
state_embed = StateEmbed(configurer=state_config)

adata = sc.read_h5ad("example.h5ad")

processed_data = state_embed.process_data(adata=adata)
embeddings = state_embed.get_embeddings(processed_data)

```

**Example State Transitions Usage:**

See below for the steps to run the STATE transition model for perturbing cells. The notebook shows a full example with example input data.

```python
from helical.models.state import StatePerturb
from helical.models.state import StateConfig
import scanpy as sc
import random 

# see the notebook for an example perturbations added to the data
adata = sc.read_h5ad("example_data.h5ad")
perturbations = [
    pert_1,
    pert_2,
    pert_3,
]

adata.obs['target_gene'] = random.choices(perturbations, k=n_cells)
perturb_config = StateConfig()
state_perturb = StatePerturb(configurer=perturb_config)

processed_data = state_perturb.process_data(adata)
perturbed_embeds = state_perturb.get_embeddings(processed_data)
```

## Citation

To cite the STATE model, please use the following reference:
```bibtex
 @article{Adduri_2025, title={Predicting cellular responses to perturbation across diverse contexts with State}, url={http://dx.doi.org/10.1101/2025.06.26.661135}, DOI={10.1101/2025.06.26.661135}, publisher={Cold Spring Harbor Laboratory}, author={Adduri, Abhinav K. and Gautam, Dhruv and Bevilacqua, Beatrice and Imran, Alishba and Shah, Rohan and Naghipourfar, Mohsen and Teyssier, Noam and Ilango, Rajesh and Nagaraj, Sanjay and Dong, Mingze and Ricci-Tam, Chiara and Carpenter, Christopher and Subramanyam, Vishvak and Winters, Aidan and Tirukkovular, Sravya and Sullivan, Jeremy and Plosky, Brian S. and Eraslan, Basak and Youngblut, Nicholas D. and Leskovec, Jure and Gilbert, Luke A. and Konermann, Silvana and Hsu, Patrick D. and Dobin, Alexander and Burke, Dave P. and Goodarzi, Hani and Roohani, Yusuf H.}, year={2025}, month=jun }
```

For more details and updates, visit the [STATE GitHub repository](https://github.com/ArcInstitute/state).

