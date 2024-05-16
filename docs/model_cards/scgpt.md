# Model Card for scGPT

## Model Details

**Model Name:** scGPT  
**Model Version:** 1.0  
**Model Description:** scGPT is a generative pre-trained transformer model for single-cell multi-omics analysis. It is designed to perform various tasks, including cell type annotation, multi-batch integration, multi-omic integration, perturbation response prediction, and gene network inference. The model is pre-trained on extensive single-cell RNA sequencing data to build a foundational understanding of cellular biology.

## Model Developers

**Developed By:** Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan, Bo Wang  
**Contact Information:** Bo Wang (bowang@vectorinstitute.ai)  
**License:** MIT License

## Model Type

**Architecture:** Transformer-based  
**Domain:** Cell Biology, Bioinformatics  
**Languages:** Single-cell transcriptomics data

## Model Purpose

**Intended Use:**  
- Research in single-cell genomics and bioinformatics
- Cell type annotation and data integration in single-cell studies
- Educational purposes in the context of multi-omics data analysis

**Out-of-Scope Use Cases:**  
- Direct clinical decision making without human oversight
- Applications outside the scope of single-cell multi-omics analysis

## Training Data

**Data Sources:**  
- Publicly available single-cell RNA-seq, ATAC-seq, and other omics databases from CELLxGENE and other repositories

**Data Volume:**  
- Pre-trained on data from over 33 million single-cell samples

**Preprocessing:**  
- Standardized to remove low-quality cells and sequences
- Normalization and scaling to ensure consistency across datasets

## Model Performance

**Evaluation Metrics:**  
- Accuracy, Precision, Recall, F1-Score for tasks like cell type annotation, data imputation, and perturbation response prediction

**Performance Benchmarks:**  
- Cell Type Annotation: Precision > 0.8 for most cell types
- Perturbation Prediction: Outperformed other methods in predicting post-perturbation changes with significant margins

**Testing Data:**  
- Held-out subsets of the training dataset
- Additional external validation datasets from independent studies

## Ethical Considerations

**Bias and Fairness:**  
- Ensured diverse representation of cell types and conditions in the training data
- Ongoing evaluation for biases, particularly those impacting underrepresented cell types or conditions

**Privacy:**  
- All training data sourced from public databases with appropriate usage permissions
- No use of private or sensitive genetic data without explicit consent

**Mitigations:**  
- Regular audits of model outputs to detect and correct biases
- Collaboration with ethicists and domain experts to ensure responsible use

## Model Limitations

**Known Limitations:**  
- May not generalize well to rare cell types or novel conditions not represented in the training data
- Performance may vary across different sequencing technologies and experimental conditions

**Future Improvements:**  
- Continuous integration of new data sources and modalities
- Enhancements in model architecture to better handle rare cell types and novel conditions

## How to Use

**Input Format:**  
- Standard single-cell RNA-seq, ATAC-seq, and other omics data formats (e.g., CSV, H5AD)

**Output Format:**  
- JSON format with predicted cell types, imputed data, and integrated multi-modal data

**Example Usage:**
```python
from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad

model_config = scGPTConfig(batch_size=10)
scgpt = scGPT(model_config=model_config)

adata = ad.read_h5ad("dataset.h5ad")
data = scgpt.process_data(adat)
embeddings = scgpt.get_embeddings(data)

print(embeddings.shape)
```

## Developers

- Haotian Cui
- Chloe Wang
- Hassaan Maan
- Kuan Pang
- Fengning Luo
- Nan Duan
- Bo Wang

## Contact

- Bo Wang (bowang@vectorinstitute.ai)

## Citation

To cite the scGPT model, please use the following reference:
```
@article{cui2023scGPT,
  title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
  author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

For more details and updates, visit the [scGPT GitHub repository](https://github.com/bowang-lab/scGPT).