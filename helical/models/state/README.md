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

- Tokenizing transcriptomes
- Tokenizing conditions (i.e. meta-information associated with individual genes, like perturbation experiment alterations, which are indicated by perturbation tokens)
- Pretraining
- Finetuning 
- Pre- and post-perturbation cell embeddings

**Broader research applications:**  

- Perturbation response prediction

## Training Data

**Data Sources:** 

- Publicly available datasets and code are described in the preprint's code and model availability section.

**Data Volume:**  

- Pre-trained on data from 167 million human cells for the embedding model and over 100 million cells for the transition model. The perturbed cells are chemically or genetically perturbed cells from large scale single-cell screens.

**Preprocessing:**  

- Normalization and scaling to ensure consistency across datasets

## Model Performance

**Evaluation Metrics:**  

- Classification metrics: 
- Biological conservation metrics: 
- Batch correction metrics: 

**Testing Data:**  

- Held-out subsets of the training dataset

## Model Limitations

**Known Limitations:**

- An inherent limitations of single-cell RNA sequencing data requires the destruction of cells during measurement. This prevents observations of their non-perturbed states. Some perturbations such as gene knockout may also not occur experimentally and incorrectly flagged as a pertubed datapoint. 

- STATE evaluation metrics are more sensitive on an individual gene basis but stronger on an ensemble level. As pointed out by the authors metrics such as accuracy of individual DE genes depends on dataset size and quality.

- Attention maps are sensitive to cell set heterogeneity.

**Future Improvements:**  

- Improving perturbation featurisation between modalities such as drugs and gene knockdown, which would allow for combinatorial perturbation predictions.

- STATE has been trained on over 70 cell contexts but not tested for cell context not completely seen during training (e.g., unseen cell types).

- Training of the STATE embedding model on larger datasets.

## How to Use

**Input Format:**  

- Annotated data matrix (.anndata)

**Output Format:**  

- Cell embeddings


**Example State Embedding Usage:**
```python

from helical.models.state import stateEmbeddingsModel
from helical.models.state import stateConfig

state_config = stateConfig()

state_embed = stateEmbeddingsModel(configurer = state_config)
processed_data = state_embed.process_data(ann_data_path="path/to/data.h5ad")
embeddings = state_embed.get_embeddings(processed_data)

```

**Example State Transitions Usage:**

See below for the steps to run the STATE transition model for perturbing cells. We show a more concrete example shortly for the Virtual Cell Challenge data with input data.

```python
from helical.models.state import stateTransitionModel
from helical.models.state import stateConfig
import scanpy as sc

state_config = stateConfig()

state_transition = stateTransitionModel(configurer=state_config)

adata = sc.read_h5ad("example_data.h5ad")
adata = state_transition.process_data(adata)
adata = state_transition.get_embeddings(adata)
```

**Example State Transition Finetuning:**

We can add a classification or regression head to the perturbed cell embeddings as below.

```python
from helical.models.state import stateFineTuningModel

scgpt_fine_tune = stateFineTuningModel(configurer = state_config, fine_tuning_head = "classification", output_size = 2) 
data = scgpt_fine_tune.process_data("input_dict")
scgpt_fine_tune.train()
```

**Creasting a Virtual Cell Challenge Submission:**

Similar to the Colab code by the authors we download the relevant datasets.

```python
'''
Download the dataset

(taken from Colab Notebook by Adduri et al.
https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l#scrollTo=h0aSjKX7Rtyw)
'''

import requests
from tqdm.auto import tqdm  # picks the best bar for the environment
from zipfile import ZipFile
from tqdm.auto import tqdm
import os

# Download the Replogle-Nadig training dataset.
url = "https://storage.googleapis.com/vcc_data_prod/datasets/state/competition_support_set.zip"
output_path = "competition_support_set.zip"

# stream the download so we can track progress
response = requests.get(url, stream=True)
total = int(response.headers.get("content-length", 0))

with open(output_path, "wb") as f, tqdm(
    total=total, unit='B', unit_scale=True, desc="Downloading"
) as bar:
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            break
        f.write(chunk)
        bar.update(len(chunk))

out_dir  = "competition_support_set"
os.makedirs(out_dir, exist_ok=True)
with ZipFile(output_path, 'r') as z:
    for member in tqdm(z.infolist(), desc="Unzipping", unit="file"):
        z.extract(member, out_dir)
```

Once downloaded, edit the competition_support_set/starter.toml to point to the correct dataset path at the top. 

We can now train the model on the dataset with the below code.

```python

from helical.models.state import stateTransitionTrainModel
from helical.models.state.train_configs import trainingConfig

train_config = trainingConfig(
    toml_config_path="competition_support_set/starter.toml",
    num_workers=4,
    batch_col="batch_var",
    pert_col="target_gene",
    cell_type_key="cell_type",
    control_pert="non-targeting",
    perturbation_features_file="competition_support_set/ESM2_pert_features.pt",
    max_steps=40000,
    ckpt_every_n_steps=20000,
    model="state")

state_train = stateTransitionTrainModel(configurer = train_config)
state_train.train() 
state_train.predict() 

```

We use the `stateTransitionTrainModel` class when wanting to train on new datasets. Once complete, checkpoint files will be generated that can be used to initialise the `stateTransitionModel` class for inference.

We can now use the model for inference on the validation dataset 

```python
from helical.models.state import stateConfig
from helical.models.state import stateTransitionModel
import scanpy as sc

state_config = stateConfig(
    output = "competition/prediction.h5ad",
    model_dir = "competition/first_run",
    checkpoint = "competition/first_run/checkpoints/final.ckpt",
    pert_col = "target_gene",
    embed_key = None,
    celltype_col = None,
    celltypes = None,
    batch_col = None,
    control_pert = None,
    seed = 42,
    max_set_len = None,
    tsv = None
)

adata = sc.read_h5ad("competition_support_set/competition_val_template.h5ad")

state_transition = stateTransitionModel(configurer=state_config)
adata = state_transition.process_data(adata)
embeds = state_transition.get_embeddings(adata)
```

The code will generate a .vcc file for predictions which can be evaluated using the cell-eval package for submission to the public Virtual Cell Challenge leaderboard.

```python
# evaluate the model - underlying function uses cell-eval package 
# (https://github.com/ArcInstitute/cell-eval)
from helical.models.state import vcc_eval

# default configs for competition dataset
EXPECTED_GENE_DIM = 18080
MAX_CELL_DIM = 100000
DEFAULT_PERT_COL = "target_gene"
DEFAULT_CTRL = "non-targeting"
DEFAULT_COUNTS_COL = "n_cells"
DEFAULT_CELLTYPE_COL = "celltype"
DEFAULT_NTC_NAME = "non-targeting"

configs = {
    # path to the prediction file
    "input": "competition/prediction.h5ad",
    # path to the gene names file
    "genes": "competition_support_set/gene_names.csv",
    # path to the output file - if None will be created with default naming
    "output": None,
    "pert_col": DEFAULT_PERT_COL,
    "celltype_col": None,
    "ntc_name": DEFAULT_NTC_NAME,
    "output_pert_col": DEFAULT_PERT_COL,
    "output_celltype_col": DEFAULT_CELLTYPE_COL,
    "encoding": 32,
    "allow_discrete": False,
    "expected_gene_dim": EXPECTED_GENE_DIM,
    "max_cell_dim": MAX_CELL_DIM,
}

# this creates a submission file in the output directory which can be uploaded to the challenge leaderboard
vcc_eval(configs)
```









## Citation

To cite the STATE model, please use the following reference:
```bibtex
 @article{Adduri_2025, title={Predicting cellular responses to perturbation across diverse contexts with State}, url={http://dx.doi.org/10.1101/2025.06.26.661135}, DOI={10.1101/2025.06.26.661135}, publisher={Cold Spring Harbor Laboratory}, author={Adduri, Abhinav K. and Gautam, Dhruv and Bevilacqua, Beatrice and Imran, Alishba and Shah, Rohan and Naghipourfar, Mohsen and Teyssier, Noam and Ilango, Rajesh and Nagaraj, Sanjay and Dong, Mingze and Ricci-Tam, Chiara and Carpenter, Christopher and Subramanyam, Vishvak and Winters, Aidan and Tirukkovular, Sravya and Sullivan, Jeremy and Plosky, Brian S. and Eraslan, Basak and Youngblut, Nicholas D. and Leskovec, Jure and Gilbert, Luke A. and Konermann, Silvana and Hsu, Patrick D. and Dobin, Alexander and Burke, Dave P. and Goodarzi, Hani and Roohani, Yusuf H.}, year={2025}, month=jun }
```

For more details and updates, visit the [STATE GitHub repository](https://github.com/ArcInstitute/state).

