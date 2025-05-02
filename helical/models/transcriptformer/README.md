# Model Card for TranscriptFormer

## Model Details

**Model Name:** TranscriptFormer

**Model Version:** 1.0  

**Model Description:** 
TranscriptFormer is a family of generative foundation models representing a cross-species generative cell atlas trained on up to 112 million cells spanning 1.53 billion years of evolution across 12 species. The models include three distinct versions:

- **TF-Metazoa**: Trained on 112 million cells spanning all twelve species. The set covers six vertebrates (human, mouse, rabbit, chicken, African clawed frog, zebrafish), four invertebrates (sea urchin, C. elegans, fruit fly, freshwater sponge), plus a fungus (yeast) and a protist (malaria parasite).
The model includes 444 million trainable parameters and 633 million non-trainable
parameters (from frozen pretrained embeddings). Vocabulary size: 247,388.

- **TF-Exemplar**: Trained on 110 million cells from human and four model organisms: mouse (M. musculus), zebrafish (D. rerio), fruit fly (D. melanogaster ), and C. ele-
gans. Total trainable parameters: 542 million; non-trainable: 282 million. Vocabulary size:
110,290.

- **TF-Sapiens**: Trained on 57 million human-only cells. This model has 368 million trainable parameters and 61 million non-trainable parameters. Vocabulary size: 23,829.


TranscriptFormer is designed to learn rich, context-aware representations of single-cell transcriptomes while jointly modeling genes and transcripts using a novel generative architecture. It employs a generative autoregressive joint model over genes and their expression levels per cell across species, with a transformer-based architecture, including a novel coupling between gene and transcript heads, expression-aware multi-head self-attention, causal masking, and a count likelihood to capture transcript-level variability. TranscriptFormer demonstrates robust zero-shot performance for cell type classification across species, disease state identification in human cells, and prediction of cell type specific transcription factors and gene-gene regulatory relationships. This work establishes a powerful framework for integrating and interrogating cellular diversity across species as well as offering a foundation for in-silico experimentation with a generative single-cell atlas model.

For more details, please refer to the manuscript: [A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution: The TranscriptFormer Single-cell Model](https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1)


## Model Developers

**Developed By:** James D Pearce, Sara E Simmonds*, Gita Mahmoudabadi*, Lakshmi Krishnan*, Giovanni
Palla, Ana-Maria Istrate, Alexander Tarashansky, Benjamin Nelson, Omar Valenzuela,
Donghui Li, Stephen R Quake, Theofanis Karaletsos (Chan Zuckerberg Initiative)

*Equal contribution



**License:** MIT License

## Model Type

**Architecture:** Transformer-based  

**Domain:** Cell Biology, Bioinformatics  

**Languages:** Single-cell transcriptomics data 

## Model Purpose

**Use cases:**

- **Integrating cellular diversity across species:** 

	Enables cross-species comparisons to better understand conserved and divergent cell types over evolutionary time.

- **Interrogating cellular diversity:**

	Facilitates in-depth analysis of cell states, types, and transitions within and across organisms.

- **In silico exploration:**

	Supports computational hypothesis generation, such as predicting gene expression changes or cell fate trajectories without additional wet-lab experiments.

- **Foundation for biological discovery:**

	Acts as a generative tool for designing experiments, exploring perturbations, and guiding downstream analyses in a virtual setting.

## Training Data

**Data Sources:**
Publicly available datasets are described in `Data availability` in the [manuscript](https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1)


**Total size:**
Up to 112 million cells spanning 1.53 billion years of evolution
across 12 species

## Model Limitations

**Known Limitations:**  
- Similar to most foundtation models, batch effects are not yet handled in a unified way.
- TranscriptFormer is not specialized for zero-shot perturbation prediction.

**Future Improvements:**  
- Expand species diversity in training data to better capture evolutionary variation in cellular biology.
- Incorporate additional biological modalities for multimodal understanding.

## How to Use

**Example Usage:**

```python
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import TranscriptFormerConfig
import anndata as ad

configurer = TranscriptFormerConfig()
model = TranscriptFormer(configurer)

ann_data = ad.read_h5ad("/path/to/data.h5ad")

dataset = model.process_data([ann_data])
embeddings = model.get_embeddings(dataset)
print(embeddings)
```

**Example Fine-Tuning**


Coming soon!

## Contact
Theofanis Karaletsos (tkaraletsos@chanzuckerberg.com) and Stephen
R. Quake (steve@quake-lab.org)

## Citation

```bibtex
@article {Pearce2025.04.25.650731,
	author = {Pearce, James D and Simmonds, Sara E and Mahmoudabadi, Gita and Krishnan, Lakshmi and Palla, Giovanni and Istrate, Ana-Maria and Tarashansky, Alexander and Nelson, Benjamin and Valenzuela, Omar and Li, Donghui and Quake, Stephen R. and Karaletsos, Theofanis},
	title = {A Cross-Species Generative Cell Atlas Across 1.5 Billion Years of Evolution: The TranscriptFormer Single-cell Model},
	elocation-id = {2025.04.25.650731},
	year = {2025},
	doi = {10.1101/2025.04.25.650731},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/04/29/2025.04.25.650731},
	eprint = {https://www.biorxiv.org/content/early/2025/04/29/2025.04.25.650731.full.pdf},
	journal = {bioRxiv}
}
```