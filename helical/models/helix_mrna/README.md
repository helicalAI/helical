# Model Card for Helix-mRNA

## Model Details

**Model Name:** Helix-mRNA  

**Model Versions:** v0 

**Model Description:** Helix-mRNA is a single nucleotide resolution model that combines the Mamba2 architecture with transformer components, including attention and MLP blocks. The hybrid architecture enables precise nucleotide-level analysis and prediction of mRNA sequences. By leveraging both the efficient sequence processing capabilities of Mamba2's state space architecture and the contextual understanding of transformer attention mechanisms, Helix-mRNA processes mRNA sequences at individual nucleotide resolution. The model incorporates a special 'E' character to denote the beginning of each codon, enhancing its ability to recognize and analyze codon-level patterns in mRNA sequences.

## Model Developers

**Developed By:** 
Helical Team 

**Contact Information:** 
hello@helical-ai.com

**License:** 
CC-BY-NC-SA 4.0 

## Model Type

**Architecture:** 
Mamba2-Transformer hybrid

**Domain:** 
mRNA Bioinformatics  

**Input Data:** 

- mRNA Sequence Data (A, C, U, G, N and E)
- E is used to denote the beginning of each codon

## Model Purpose

**Intended Use:**  

- Tokenizing mRNA sequences at single-nucleotide resolution
- Generating rich embeddings for embedding and plotting
- Pretraining
- Downstream task fine-tuning

**Use Cases**

- mRNA Optimization
  - Translation efficiency improvement
  - Stability enhancement
  - Half-life modification
  - Sequence evaluation and candidate prioritization

- Therapeutic Applications
  - mRNA vaccine design (SARS-CoV-2)
  - Personalized cancer vaccines
  - Neoantigen encoding

- Biomanufacturing
  - Industrial-scale mRNA synthesis optimization
  - Yield improvement
  - Production consistency enhancement
  - Biologics manufacturing

## Training Data

**Pretraining:**  

- The pretraining dataset comprises 57.5 million mRNA sequences across five taxonomic groups: Other Vertebrates, Mammals, Invertebrates, Fungi, and Human-Host Viral sequences for genomic and evolutionary analyses.

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
- Supports standard mRNA sequence formats (A, U, G, C) and the E character to denote the beginning of codons

**Output Format** 
- Nucleotide-level embeddings

**Example Usage**

```python
from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

helix_mrna_config = HelixmRNAConfig(batch_size=5, max_length=100, device=device)
helix_mrna = HelixmRNA(configurer=helix_mrna_config)

rna_sequences = ["EACUEGGG", "EACUEGGG", "EACUEGGG", "EACUEGGG", "EACUEGGG"]
dataset = helix_mrna.process_data(rna_sequences)

rna_embeddings = helix_mrna.get_embeddings(dataset)

print("Helix-mRNA embeddings shape: ", rna_embeddings.shape)
```

**Example Fine-Tuning:**

```python
from helical.models.helix_mrna import HelixmRNAFineTuningModel, HelixmRNAConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

input_sequences = ["EACU"*20, "EAUG"*20, "EUGC"*20, "ECUG"*20, "EUUG"*20]
labels = [0, 2, 2, 0, 1]

helix_mrna_config = HelixmRNAConfig(batch_size=5, device=device, max_length=100)
helix_mrna_fine_tune = HelixmRNAFineTuningModel(helix_mrna_config=helix_mrna_config, output_size=3)

train_dataset = helix_mrna_fine_tune.process_data(input_sequences)

helix_mrna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

outputs = helix_mrna_fine_tune.get_outputs(train_dataset)

print("Helix-mRNA fine-tuned model output shape", outputs.shape)
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