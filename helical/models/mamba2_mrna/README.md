# Model Card for Mamba2-mRNA

## Model Details

**Model Name:** Mamba2-mRNA  

**Model Versions:** 1.0  

**Model Description:** Mamba2-mRNA is a single nucleotide resolution model built using the Mamba2 architecture. The model employs 16 Mamba layers (16L) to enable precise nucleotide-level analysis and prediction of mRNA sequences. By leveraging the efficient sequence processing capabilities of Mamba2's state space architecture, Mamba2-mRNA can process mRNA sequences at individual nucleotide resolution, making it suitable for detailed mRNA sequence analysis tasks.

## Model Developers

**Developed By:** 
Helical Team 

**Contact Information:** 
support@helical-ai.com

**License:** 
CC-BY-NC-SA 4.0 

## Model Type

**Architecture:**
Mamba2-based  

**Domain:** 
mRNA Bioinformatics 

**Input Data:** 
mRNA Sequence Data (A, C, U, G and N)

## Model Purpose

**Intended Use:**  

- Tokenizing mRNA sequences at single-nucleotide resolution
- Generating rich embeddings for embedding and plotting
- Pretraining
- Downstream task fine-tuning

## Training Data

**Data Sources:**  

- Publicly available mRNA sequence databases

**Preprocessing:**  

- Check that mRNA sequences only contains valid input characters

## Model Performance

**Evaluation Metrics:**  

- Accuracy, Precision, Pearson, Spearmanr, ROC AUC

## Model Limitations

**Known Limitations:**  
- Model performance may vary based on sequence length.
- Model performance degrades for sequences longer than those used during pretraining.
- Specific to mRNA sequence analysis tasks.

## How to Use

**Input Format** 

- mRNA sequences at nucleotide-level resolution.
- Supports standard mRNA sequence formats (A, U, G, C)

**Output Format** 

- Nucleotide-level embeddings

**Example Usage**

```python
from helical.models.mamba2_mrna import Mamba2mRNA, Mamba2mRNAConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

input_sequences = ["ACUG"*20, "AUGC"*20, "AUGC"*20, "ACUG"*20, "AUUG"*20]

mamba2_mrna_config = Mamba2mRNAConfig(batch_size=5, device=device)
mamba2_mrna = Mamba2mRNA(configurer=mamba2_mrna_config)

processed_input_data = mamba2_mrna.process_data(input_sequences)

embeddings = mamba2_mrna.get_embeddings(processed_input_data)

print("Mamba2-mRNA embeddings shape: ", embeddings.shape)
```

**Example Fine-Tuning:**

```python
from helical.models.mamba2_mrna import Mamba2mRNAFineTuningModel, Mamba2mRNAConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

input_sequences = ["ACUG"*20, "AUGC"*20, "AUGC"*20, "ACUG"*20, "AUUG"*20]
labels = [0, 2, 2, 0, 1]

mamba2_mrna_config = Mamba2mRNAConfig(batch_size=5, device=device, max_length=100)
mamba2_mrna_fine_tune = Mamba2mRNAFineTuningModel(mamba2_mrna_config=mamba2_mrna_config, output_size=3)

train_dataset = mamba2_mrna_fine_tune.process_data(input_sequences)

mamba2_mrna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

outputs = mamba2_mrna_fine_tune.get_outputs(train_dataset)

print("Mamba2-mRNA fine-tuned model output shape", outputs.shape)
```

## Contact
support@helical-ai.com

## Citation
```bibtex
@software{allard_2024_13135902,
  author       = {Helical Team},
  title        = {helicalAI/helical: v0.0.1a14},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.1a14},
  doi          = {10.5281/zenodo.13135902},
  url          = {https://doi.org/10.5281/zenodo.13135902}
}
```