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

# Get attention weights (requires attn_impl='torch')
tahoe_config_attn = TahoeConfig(
    model_size="70m",
    batch_size=8,
    attn_impl="torch"  # Use 'torch' instead of 'flash' for attention extraction
)
tahoe_attn = Tahoe(configurer=tahoe_config_attn)
dataloader_attn = tahoe_attn.process_data(adata)
cell_embeddings, attentions = tahoe_attn.get_embeddings(
    dataloader_attn,
    output_attentions=True
)
```

## Features

- **Self-contained**: No need to install or clone the separate tahoe-x1 package
- **Minimal dependencies**: Only ~2,315 lines from tahoe-x1 (16% reduction through cleanup)
- **Clean API**: Clear separation between data processing and embedding extraction
- **Follows helical patterns**: Uses the same structure as other models (Geneformer, scGPT)
- **Automatic gene mapping**: Maps gene symbols to Ensembl IDs using helical utilities
- **Flexible embeddings**: Supports both cell-level and gene-level embeddings
- **Attention extraction**: Supports attention weight extraction when using `attn_impl='torch'`
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

## Attention Implementation

The model supports two attention implementations:

### Flash Attention (default)
- **Fast and memory efficient**: Optimized for speed and reduced memory usage
- **Default setting**: `attn_impl='flash'`
- **Limitation**: Does not support attention weight extraction
- **Best for**: Production inference and large-scale embedding extraction

### Standard PyTorch Attention
- **Slower but flexible**: Uses standard PyTorch attention mechanism
- **Enable with**: `attn_impl='torch'`
- **Supports**: Attention weight extraction for analysis and visualization
- **Best for**: Research and analysis requiring attention maps

Example:
```python
# For standard inference (fast)
config = TahoeConfig(model_size="70m", attn_impl="flash")

# For attention extraction (slower)
config = TahoeConfig(model_size="70m", attn_impl="torch")
tahoe = Tahoe(configurer=config)
embeddings, attentions = tahoe.get_embeddings(dataloader, output_attentions=True)
```

## Model Details

The Tahoe-1x model is a transformer-based foundation model for single-cell RNA-seq analysis:
- Uses Ensembl IDs for gene identification
- Supports human genes
- Available from Hugging Face: `tahoebio/Tahoe-x1`
- Three model sizes: 70M, 1B, and 3B parameters
