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
- Tokenizing conditions (i.e. perturbation experiment alterations, which are indicated by perturbation tokens)
- Pretraining 
- Finetuning 
- Pre- and post-perturbation cell embeddings

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
- Biological conservation metrics: Cell-Eval (incl. p-values, fold change, ranking, correlation)
- Batch correction metrics: Batch Embeddings added to input

**Testing Data:**  

- Held-out subsets of the training dataset

## Model Limitations

**Known Limitations:**

- Single-cell RNA sequencing data requires the destruction of cells during measurement. This prevents observations of their non-perturbed states. Some perturbations such as gene knockout may also not occur experimentally and incorrectly flagged as a pertubed datapoint. 

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

- Cell embeddings


**Example State Embedding Usage:**
```python

from helical.models.state import stateEmbeddingsModel
from helical.models.state import stateConfig
import scanpy as sc

state_config = stateConfig()
state_embed = stateEmbeddingsModel(configurer=state_config)

adata = sc.read_h5ad("example.h5ad")

processed_data = state_embed.process_data(adata=adata)
embeddings = state_embed.get_embeddings(processed_data)

```

**Example State Transitions Usage:**

See below for the steps to run the STATE transition model for perturbing cells. We show a more concrete example shortly for the Virtual Cell Challenge with input data.

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
from helical.models.state import stateModularFineTuningModel
from helical.models.state import stateConfig
import scanpy as sc

# Load the desired dataset
adata = sc.read_h5ad("example.h5ad")

# Get the desired label class
cell_types = list(adata.obs.cell_type)

# Get unique labels
label_set = set(cell_types)

# Create the fine-tuning model with the relevant configs
config = stateConfig()
model = stateModularFineTuningModel(
    configurer=config, 
    fine_tuning_head="classification", 
    output_size=len(label_set),
)

# Process the data for training 
data = model.process_data(adata)

# Create a dictionary mapping the classes to unique integers for training
class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

for i in range(len(cell_types)):
    cell_types[i] = class_id_dict[cell_types[i]]

print(f"Converted {len(cell_types)} labels to integers")

# Fine-tune
model.train(train_input_data=data, train_labels=cell_types)
```

**Creating a Virtual Cell Challenge Submission:**

Similar to the Colab notebook by the authors we download the relevant datasets.

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

Once downloaded, edit `competition_support_set/starter.toml` to point to the correct dataset path (the top line). 
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

We use the `stateTransitionTrainModel` class when wanting to train on new datasets. Once complete, checkpoint files will be generated that can be used to initialise the `stateTransitionModel` class for future inference.

We run model inference using the new checkpoint on the validation dataset. 

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

This will generate a `.vcc` predictions file which can be evaluated using the `cell-eval` package for submission to the public Virtual Cell Challenge leaderboard.

```python

```

## Citation

To cite the STATE model, please use the following reference:
```bibtex
 @article{Adduri_2025, title={Predicting cellular responses to perturbation across diverse contexts with State}, url={http://dx.doi.org/10.1101/2025.06.26.661135}, DOI={10.1101/2025.06.26.661135}, publisher={Cold Spring Harbor Laboratory}, author={Adduri, Abhinav K. and Gautam, Dhruv and Bevilacqua, Beatrice and Imran, Alishba and Shah, Rohan and Naghipourfar, Mohsen and Teyssier, Noam and Ilango, Rajesh and Nagaraj, Sanjay and Dong, Mingze and Ricci-Tam, Chiara and Carpenter, Christopher and Subramanyam, Vishvak and Winters, Aidan and Tirukkovular, Sravya and Sullivan, Jeremy and Plosky, Brian S. and Eraslan, Basak and Youngblut, Nicholas D. and Leskovec, Jure and Gilbert, Luke A. and Konermann, Silvana and Hsu, Patrick D. and Dobin, Alexander and Burke, Dave P. and Goodarzi, Hani and Roohani, Yusuf H.}, year={2025}, month=jun }
```

For more details and updates, visit the [STATE GitHub repository](https://github.com/ArcInstitute/state).

