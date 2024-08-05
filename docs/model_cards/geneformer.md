# Model Card for Geneformer

## Model Details

**Model Name:** Geneformer  \
**Model Version:** 1.0  \
**Model Description:** Geneformer is a context-aware, attention-based deep learning model pretrained on a large-scale corpus of approximately 30 million single-cell transcriptomes. It is designed to enable context-specific predictions in settings with limited data in network biology. The model performs various tasks such as gene network mapping, disease modeling, and therapeutic target identification. 

## Model Developers

**Developed by:** Christina V. Theodoris conceived of the work, developed Geneformer, assembled Genecorpus-30M and designed and performed computational analyses. Other [author contributions](#citation). \
**Contact Information:** christina.theodoris@gladstone.ucsf.edu  \
**License:** Apache-2.0 

## Model Type

**Architecture:** Transformer-based \
**Domain:** Cell Biology, Bioinformatics  \
**Input Data:** Single-cell transcriptomes\

## Model Purpose

**Technical usage:**  
- Tokenizing transcriptomes
- Pre-training
- Hyperparameter tuning
- Fine-tuning
- Extracting and plotting cell embeddings
- In silico perturbation


**Broader research applications:**  
- Research in genomics and network biology
- Disease modeling with limited patient data
- Identification of candidate therapeutic targets
- Prediction of gene dosage sensitivity and chromatin dynamics
- Context-specific predictions in gene regulatory networks

**Out-of-Scope Use Cases:**  
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
- Rank value encoding of transcriptomes where genes are ranked by scaled expression within each cell.

## Model Performance

**Evaluation Metrics:**  
- Area Under the Receiver Operating Characteristic Curve (AUC)
- Predictive accuracy in distinguishing:
    - With *fine-tuning*:
        - Transcription factor dosage sensitivity
        - Chromatin dynamics (bivalently marked promoters)
        - Transcription factor regulatory range
        - Gene network centrality
        - Transcription factor targets
        - Cell type annotation
        - Batch integration
        - Cell state classification across differentiation
        - Disease classification
        - In silico perturbation to determine disease-driving genes
        - In silico treatment to determine candidate therapeutic targets
    - With *Zero-shot learning*:
        - Batch integration
        - Gene context specificity
        - In silico reprogramming
        - In silico differentiation
        - In silico perturbation to determine impact on cell state
        - In silico perturbation to determine transcription factor targets
        - In silico perturbation to determine transcription factor cooperativity

**Testing Data:**  
- Held-out subsets of the training dataset
- Additional validation using publicly available datasets 
- Experimental validation for: 
    - Prediction of novel transcription factor in cardiomyocytes with zero-shot learning that had a functional impact on cardiomyocytes' contractile force generation 
    - Prediction of candidate therapeutic targets with in silico treatment analysis that significantly improved contractile force generation of cardiac microtissues in an iPS cell model of cardiomyopathy

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

model_config = GeneformerConfig(batch_size = 10)
geneformer = Geneformer(model_config = model_config)
ann_data = ad.read_h5ad("dataset.h5ad")
dataset = geneformer.process_data(ann_data)
embeddings = geneformer.get_embeddings(dataset)

print(embeddings.shape)

```

## Contact

christina.theodoris@gladstone.ucsf.edu

## Citation

Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., Mantineo, H., Brydon, E. M., Zeng, Z., Liu, X. S., & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. Nature, 618, 616-624. https://doi.org/10.1038/s41586-023-06139-9

## Author contributions 

C.V.T. conceived of the work, developed Geneformer, assembled Genecorpus-30M and designed and performed computational analyses. L.X., A.C., Z.R.A.S., M.C.H., H.M. and E.M.B. performed experimental validation in engineered cardiac microtissues. M.D.C. performed preprocessing, cell annotation and differential expression analysis of the cardiomyopathy dataset. Z.Z. provided data from the TISCH database for inclusion in Genecorpus-30M. X.S.L. and P.T.E. designed analyses and supervised the work. C.V.T., X.S.L. and P.T.E. 
