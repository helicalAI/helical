# Model Card for Universal Cell Embedding (UCE)

## Model Details

**Model Name:** Universal Cell Embedding (UCE) \
**Model Version:** 1.0 \
**Model Description:** A large-scale self-supervised transformer-based model, pre-trained across more than 36 million cells for creating universal representations of cell types across different species and tissues. The model enables cell type annotation prediction, hypothesis generation, disease state comparison, new data mapping, integration of diverse single-cell datasets, and opens the door for discovery of novel cell type functions.

## Model Developers

**Developed By:** Yanay Rosen and Yusuf Roohani conceived the study, performed research, contributed to new analytical tools, designed algorithmic frameworks, analyzed data, performed experiments and developed the software. Other [author contributions](#citation). \
**Contact Information:** jure@cs.stanford.edu, quake@stanford.edu \  
**License:** MIT License \

## Model Type

**Architecture:** Transformer-based \
**Domain:** Cell Biology, Bioinformatics \
**Input Data:** Single-cell transcriptomics data \

## Model Purpose

**Intended Use:**  
- Research in cell biology and bioinformatics
- Annotation of cell types in large single-cell datasets
- Hypothesis generation in biological research

**Out-of-Scope Use Cases:**  
- Direct clinical decision making without human oversight
- Applications outside the scope of single-cell data analysis

## Training Data

**Data Sources:**  
- Public single-cell transcriptomic datasets (e.g., CellXGene, various GEO datasets)
- Data from multiple species (including humans, mice, lemurs, zebrafish, pigs, monkeys, and frogs) and tissues to ensure diversity
- Download the full list of datasets used to train UCE [here](https://www.biorxiv.org/content/biorxiv/early/2023/11/29/2023.11.28.568918/DC3/embed/media-3.xlsx?download=true) 

**Data Volume:**  
- Trained across more than 300 datasets consisting of over 36 million cells and more than 1,000 different cell types

**Preprocessing:**  
- Standardized to remove low-quality data
- Balanced to ensure representation across species and cell types


## Model Performance

**Evaluation Metrics:**  
- Accuracy, Precision, Recall, F1-Score for cell type annotation
- Embedding quality evaluated by ability to uncover biological insights

**Performance Benchmarks:**  
- Precision and recall metrics in line with current state-of-the-art models
- Emergent behaviors such as identifying developmental lineages and novel species integration

**Testing Data:**  
- Held-out subsets of training datasets
- External validation using diverse single-cell datasets

## Ethical Considerations

**Bias and Fairness:**  
- Inclusion of diverse species and cell types to minimize bias
- Continuous evaluation for potential biases

**Privacy:**  
- Training data sourced from public datasets with appropriate usage permissions
- No private or sensitive genetic data used without consent

**Mitigations:**  
- Regular audits to identify and address biases
- Collaboration with ethicists and domain experts

## Model Limitations

**Known Limitations:**  
- May not generalize well to novel species or rare cell types not present in training data
- Performance may vary with different single-cell sequencing technologies

**Future Improvements:**  
- Ongoing integration of new datasets
- Enhancements to model architecture for better handling of rare cell types

## How to Use

**Input Format:**  
- Single-cell transcriptomics data in appropriate formats (e.g., h5ad)

**Output Format:**  
- JSON or h5ad format with cell type annotations and embeddings

**Example Usage:**
```python
from helical.models.uce.model import UCE, UCEConfig
import anndata as ad

configurer=UCEConfig(batch_size=10)
uce = UCE(configurer=configurer)
ann_data = ad.read_h5ad("dataset.h5ad")
data_loader = uce.process_data(ann_data[:10])
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)
```
- Download processed datasets used in the paper [here](https://drive.google.com/drive/folders/1f63fh0ykgEhCrkd_EVvIootBw7LYDVI7?usp=drive_link)


## Contact

For correspondence: jure@cs.stanford.edu, quake@stanford.edu

## Citation

Rosen, Y., Roohani, Y., Agarwal, A., Samotorčan, L., Tabula Sapiens Consortium, Quake, S. R., & Leskovec, J. (2023). Universal Cell Embeddings: A Foundation Model for Cell Biology. bioRxiv. https://doi.org/10.1101/2023.11.28.568918

## Author contributions 

Y.RS., Y.RH., S.Q. and J.L. conceived the study. Y.RS, Y.RH., S.Q. and J.L. performed research, contributed new analytical tools, designed algorithmic frameworks, analyzed data and wrote the manuscript. Y.RS. and Y.RH. performed experiments and developed the software. A.A. and L.S.
contributed to code and performed analyses. T.S. provided annotated data.


