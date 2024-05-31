# Model Card for Geneformer

## Model Details

**Model Name:** Geneformer  \
**Model Version:** 1.0  \
**Model Description:** Geneformer is a context-aware, attention-based deep learning model pretrained on a large-scale corpus of approximately 30 million single-cell transcriptomes. It is designed to enable context-specific predictions in settings with limited data in network biology. The model performs various tasks such as gene network mapping, disease modeling, and therapeutic target identification. \

## Model Developers

**Developed By:** Christina V. Theodoris, Ling Xiao, Anant Chopra, Mark D. Chaffin, Zeina R. Al Sayed, Matthew C. Hill, Helene Mantineo, Elizabeth M. Brydon, Zexian Zeng, X. Shirley Liu, Patrick T. Ellinor \
**Contact Information:** christina.theodoris@gladstone.ucsf.edu, ellinor@mgh.harvard.edu  \
**License:** Apache-2.0 \

## Model Type

**Architecture:** Transformer-based \
**Domain:** Cell Biology, Bioinformatics  \
**Input Data:** Single-cell transcriptomes\

## Model Purpose

**Intended Use:**  
- Research in genomics and network biology
- Disease modeling with limited patient data
- Identification of candidate therapeutic targets
- Prediction of gene dosage sensitivity and chromatin dynamics
- Context-specific predictions in gene regulatory networks

**Out-of-Scope Use Cases:**  
- Direct clinical decision making without human oversight
- Applications outside the scope of gene network and transcriptomic analysis

## Training Data

**Data Sources:**  
- Publicly available single-cell transcriptomic datasets (e.g., NCBI Gene Expression Omnibus, Human Cell Atlas, EMBL-EBI Single Cell Expression Atlas)
- Assembled from 29.9 million human single-cell transcriptomes across a wide range of tissues

**Data Volume:**  
- 29.9 million single-cell transcriptomes

**Preprocessing:**  
- Exclusion of cells with high mutational burdens
- Metrics established for scalable filtering to exclude possible doublets and/or damaged cells
- Rank value encoding of transcriptomes where genes are ranked by expression within each cell

## Model Performance

**Evaluation Metrics:**  
- Area Under the Receiver Operating Characteristic Curve (AUC)
- Predictive accuracy in distinguishing dosage-sensitive genes, chromatin dynamics, regulatory range of transcription factors, and central vs. peripheral network factors

**Performance Benchmarks:**  
- Gene Dosage Sensitivity: AUC 0.91
- Chromatin Dynamics: AUC 0.93 for bivalent vs. non-methylated, AUC 0.88 for bivalent vs. H3K4me3-only
- Regulatory Range of Transcription Factors: AUC 0.74
- Network Hierarchy Prediction: AUC 0.81

**Testing Data:**  
- Held-out subsets of the training dataset
- Additional validation using publicly available datasets and experimental validation

## Ethical Considerations

**Bias and Fairness:**  
- Ensured diverse representation of tissues in the training data
- Ongoing evaluation to detect and mitigate biases, especially those affecting specific tissue types

**Privacy:**  
- Training data sourced from public databases with appropriate usage permissions
- No use of private or sensitive genetic data without explicit consent

**Mitigations:**  
- Regular audits of model outputs to detect and correct biases
- Collaboration with ethicists and domain experts to ensure responsible use

## Model Limitations

**Known Limitations:**  
- May not generalize well to newly discovered tissues or rare gene variants
- Performance may vary across different single-cell sequencing technologies

**Future Improvements:**  
- Integration of new data sources to improve model robustness
- Enhancements in model architecture to better handle diverse transcriptomic profiles

## How to Use

**Input Format:**  
- Rank value encoded single-cell transcriptomes

**Output Format:**  
- Contextual gene and cell embeddings, contextual attention weights, and contextual predictions

**Example Usage:**
```python
from helical.models.geneformer.model import Geneformer,GeneformerConfig
import anndata as ad


model_config=GeneformerConfig(batch_size=10)
geneformer = Geneformer(model_config=model_config)

ann_data = ad.read_h5ad("dataset.h5ad")
dataset = geneformer.process_data(ann_data)

embeddings = geneformer.get_embeddings(dataset)

print(embeddings.shape)

```

## Developers

Christina V. Theodoris, Ling Xiao, Anant Chopra, Mark D. Chaffin, Zeina R. Al Sayed, Matthew C. Hill, Helene Mantineo, Elizabeth M. Brydon, Zexian Zeng, X. Shirley Liu, Patrick T. Ellinor

## Contact

christina.theodoris@gladstone.ucsf.edu, ellinor@mgh.harvard.edu

## Citation

Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., Mantineo, H., Brydon, E. M., Zeng, Z., Liu, X. S., & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. Nature, 618, 616-624. https://doi.org/10.1038/s41586-023-06139-9