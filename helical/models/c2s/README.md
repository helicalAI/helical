# Model Card for Cell2Sen (C2S-Scale)

A Helical implementation of the C2S-Scale model. We alter the original distribution (see below) by making use of the open-source weights and Gemma model and adding custom code that follows the method outlined in the original paper. The Helical implementation supports pre-processing and generating embeddings/perturbations. See the tutorial notebook for usage. 

## Model Details

**Model Name:** Cell2Sen (C2S-Scale)
**Model Version:** 1.0
**Model Description:** C2S-Scale is a large-scale foundation model for single-cell biology that transforms single-cell RNA-seq profiles into textual “cell sentences”. It leverages large language model (Gemma) architectures to unify transcriptomic and natural language data, enabling predictive and generative biological analysis. The model is trained on over **50 million human and mouse cells**, paired with textual metadata, annotations, and scientific literature, producing over one billion training tokens.
**Model Scale:** 410M → 27B parameters

## Model Developers

**Developed By:**
Syed Asad Rizvi, Daniel Levine, Aakash Patel, Shiyang Zhang, Eric Wang, Curtis J. Perry, Nicole M. Constante, Sizhuang He, David Zhang, Cerise Tang, Zhuoyang Lyu, Rayyan Darji, Chang Li, Emily Sun, David Jeong, Lawrence Zhao, Jennifer Kwan, David Braun, Brian Hafler, Hattie Chung, Rahul Dhodapkar, Bryan Perozzi, Jeffrey Ishizuka, Shekoofeh Azizi, and David van Dijk.

**Affiliations:** Yale University, Google Research, Google DeepMind, University of Southern California, Brown University.

**Contact:**

* Jeffrey Ishizuka — *[jeffrey.ishizuka@yale.edu](mailto:jeffrey.ishizuka@yale.edu)*
* Shekoofeh Azizi — *[shekazizi@google.com](mailto:shekazizi@google.com)*
* David van Dijk — *[david.vandijk@yale.edu](mailto:david.vandijk@yale.edu)*

**License:** Apache License 2.0 

---

## Model Type

**Architecture:** Decoder-only Transformer (LLM)
**Domain:** Single-cell biology, transcriptomics, multimodal bioinformatics
**Languages:** Natural language and transcriptomic “cell sentence” data

---

## Model Purpose

**Core Capabilities:**

* Predictive and generative modeling of single-cell data
* Integration of scRNA-seq, bulk RNA-seq, and textual metadata
* Fine-tuning for domain-specific tasks

**Supported Tasks:**

* Cell embeddings 
* Perturbation response prediction

---

## Training Data

**Data Sources:**

* Human Cell Atlas
* CELLxGENE database
* Associated manuscripts, metadata, and annotations

**Data Volume:**

* 50+ million single-cell transcriptomes (human and mouse)

**Preprocessing:**

* Log transformation 
* Conversion of anndata into rank-ordered “cell sentences”
* Tokenization

## Model Performance

**Evaluation Metrics:**  

- Accuracy, Gene Overlap, BERTScore, BioBERT Score, Kendall's τ, scFID

## Model Limitations

**Known Limitations:**

* Restricted to transcriptomic and textual modalities; omics data like proteomics or spatial data not yet native inputs
* Hallucinations from its LLM nature

---

## How to Use

**Input Format:**

* AnnData object containing normalized scRNA-seq data

**Output Format:**

* High-dimensional embeddings representing transcriptomic and contextual cell states

**Example: Getting Cell Embeddings**

We use the configs to set up the model. See the docstrings in `config.py` for every arguement.
Of note is the `perturbation_column`. This looks for the field in the underlying `anndata.obs`
and if not specified is set to `None`. If you forget to specify this you can always pass 
in a list later (see notebook example).

```python
from helical.models.cell2sen import Cell2Sen, Cell2SenConfig
import anndata as ad

# if you would like to use 4-bit quantization for reduced memory usage, set use_quantization=True in the config
config = Cell2SenConfig(batch_size=16, perturbation_column='perturbation')
cell2sen = Cell2Sen(configurer=config)

adata = ad.read_h5ad("dataset.h5ad")
processed_dataset = cell2sen.process_data(adata)

embeddings, attention_maps = cell2sen.get_embeddings(processed_dataset, output_attentions=True)
print("State embeddings shape:", embeddings.shape)

perturbed_dataset, perturbed_sentences = cell2sen.get_perturbations(processed_dataset)
print("Perturbed Cell Sentence:", perturbed_sentences[0])

```

## Citation

If you use *Cell2Sen (C2S-Scale)*, cite:

```bibtex
@article{rizvi2025cell2sen,
  title={Scaling Large Language Models for Next-Generation Single-Cell Analysis},
  author={Rizvi, Syed Asad and Levine, Daniel and Patel, Aakash and Zhang, Shiyang and Wang, Eric and Perry, Curtis Jamison and Constante, Nicole and He, Sizhuang and Zhang, David and Tang, Cerise and Lyu, Zhuoyang and Darji, Rayyan and Li, Chang and Sun, Emily and Jeong, David and Zhao, Lawrence and Kwan, Jennifer and Braun, David and Hafler, Brian and Chung, Hattie and Dhodapkar, Rahul M. and Perozzi, Bryan and Ishizuka, Jeffrey and Azizi, Shekoofeh and van Dijk, David},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.04.14.648850}
}
```

---

## Data and Code Availability

Code for model training is publicly available at: [Cell2Sen](https://github.com/vandijklab/cell2sentence) <br>
BioRxiv DOI: [Paper](https://doi.org/10.1101/2025.04.14.648850)
