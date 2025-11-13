# Tahoe-1x Model Integration

This directory contains the Tahoe-1x model integration for the helical library.

## Structure

```
tahoe/
├── __init__.py              # Exports Tahoe and TahoeConfig
├── model.py                 # Main Tahoe model class (self-contained embedding logic)
├── tahoe_config.py          # Configuration class for Tahoe
└── tahoe_x1/                # Minimal tahoe-x1 components
    ├── data/                # Data processing (collator, dataloader - 67 lines)
    ├── model/               # Model architecture (blocks, model)
    ├── tokenizer/           # Gene vocabulary and tokenization
    ├── utils/               # Utility functions (96 lines)
    └── loss.py              # Loss functions
```

## Copyright

All files in the `tahoe_x1/` subdirectory are:
```
Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
```

These files are extracted from the original tahoe-x1 repository and adapted for use within helical by updating import paths to use `helical.models.tahoe.tahoe_x1.*` instead of `tahoe_x1.*`.

## Usage

```python
from helical.models.tahoe import Tahoe, TahoeConfig
import anndata as ad

# Configure the model
tahoe_config = TahoeConfig(
    model_size="70m",      # Options: "70m", "1b", "3b"
    batch_size=8,
    emb_mode="cell",       # Options: "cell", "gene"
    device="cuda"          # Options: "cpu", "cuda"
)

# Initialize the model
tahoe = Tahoe(configurer=tahoe_config)

# Load and process data - returns a DataLoader
adata = ad.read_h5ad("your_data.h5ad")
dataloader = tahoe.process_data(adata)

# Get cell embeddings from the DataLoader
cell_embeddings = tahoe.get_embeddings(dataloader)

# Or get both cell and gene embeddings
cell_embeddings, gene_embeddings = tahoe.get_embeddings(
    dataloader,
    return_gene_embeddings=True
)
```

## Features

- **Self-contained**: No need to install or clone the separate tahoe-x1 package
- **Minimal dependencies**: Only ~2,315 lines from tahoe-x1 (16% reduction through cleanup)
- **Clean API**: Clear separation between data processing and embedding extraction
- **Follows helical patterns**: Uses the same structure as other models (Geneformer, scGPT)
- **Automatic gene mapping**: Maps gene symbols to Ensembl IDs using helical utilities
- **Flexible embeddings**: Supports both cell-level and gene-level embeddings
- **Model variants**: Supports 70M, 1B, and 3B parameter models from Hugging Face

## Dependencies

The model requires the following packages (specified in tahoe-x1's dependencies):
- torch
- composer
- huggingface_hub
- llmfoundry
- omegaconf
- safetensors
- scanpy
- scipy
- tqdm
- streaming (for data loading)

## Model Details

The Tahoe-1x model is a transformer-based foundation model for single-cell RNA-seq analysis:
- Uses Ensembl IDs for gene identification
- Supports human genes
- Available from Hugging Face: `tahoebio/Tahoe-x1`
- Three model sizes: 70M, 1B, and 3B parameters
