# Model Card for Helix-mRNA

## Model Details

**Model Name:** Helix-mRNA  
**Model Versions:** 1.0  
**Model Description:** Helix-mRNA is a single nucleotide resolution model built using the Mamba2 architecture. The model employs 16 Mamba layers (16L) to enable precise nucleotide-level analysis and prediction of RNA sequences. By leveraging the efficient sequence processing capabilities of Mamba2's state space architecture, Helix-mRNA can process RNA sequences at individual nucleotide resolution, making it suitable for detailed RNA sequence analysis tasks.

## Model Developers

**Developed By:** Helical Team 
**Contact Information:** hello@helical-ai.com
**License:** CC BY-NC-SA 4.0 

## Model Type

**Architecture:** Mamba2-based  
**Domain:** RNA Bioinformatics  
**Input Data:** RNA Sequence Data (A, C, U, G and N)

## Model Purpose

**Intended Use:**  
- Tokenizing RNA sequences at single-nucleotide resolution
- Generating rich embeddings for embedding and plotting
- Pretraining
- Downstream task fine-tuning

## Training Data

**Data Sources:**  
- Publicly available RNA sequence databases

<!-- **Data Volume:**  
- 1TB of raw DNA/RNA sequence data -->

**Preprocessing:**  
- Check that RNA sequences only contains valid input characters

## Model Performance

**Evaluation Metrics:**  
- Accuracy, Precision, Pearson, Spearmanr, ROC AUC

<!-- **Performance Benchmarks:**  
- Mutation Detection: Precision 0.95, Recall 0.93
- Gene Prediction: Precision 0.90, Recall 0.88

**Testing Data:**  
- Held-out subset of the training dataset
- Additional external validation datasets -->

<!-- ## Ethical Considerations

**Bias and Fairness:**  
- Ensured diverse representation of species and sequence types in the training data
- Ongoing evaluation for any biases, particularly those that may impact underrepresented species

**Privacy:**  
- All training data sourced from public databases with appropriate usage permissions
- No use of private or sensitive genetic data without explicit consent

**Mitigations:**  
- Regular audits of model outputs to detect and correct biases
- Collaboration with ethicists and domain experts to ensure responsible use -->

## Model Limitations

**Known Limitations:**  
- Model performance may vary based on sequence length.
- Specific to RNA sequence analysis tasks.

<!-- **Future Improvements:**  
- Continuous integration of new data sources
- Enhancements in model architecture to better handle rare variants -->

## How to Use

**Input Format** 
- RNA sequences at nucleotide-level resolution.
- Supports standard RNA sequence formats (A, U, G, C)

**Output Format** 
- Nucleotide-level embeddings

**Example Usage**
```python
from helical import HelixmRNA, HelixmRNAConfig

HelixmRNA = HelixmRNA(HelixmRNAConfig(batch_size=5, device='cuda'))

dataset = HelixmRNA.process_data(rna_sequence_strings_list)

embeddings = HelixmRNA.get_embeddings(dataset)
print("Helix-mRNA embeddings shape: ", embeddings.shape)
```

## How To Fine-Tune

```python
from helical import HelixmRNAFineTuningModel, HelixmRNAConfig

helixr_config = HelixmRNAConfig(batch_size=5, device='cuda')
helixr_fine_tune = HelixmRNAFineTuningModel(helixr_config=helixr_config, fine_tuning_head='regression', output_size=1)

train_dataset = helixr_fine_tune.process_data(rna_sequence_strings_list)

helixr_fine_tune.train(train_dataset=train_dataset, train_labels=output_labels)

outputs = helixr_fine_tune.get_outputs(eval_dataset)

print("Helix-mRNA fine-tuned model output shape", outputs.shape)
```


<!-- ## Developers -->

## Contact
support@helical-ai.com

## Citation
@software{allard_2024_13135902,
  author       = {Helical Team},
  title        = {helicalAI/helical: v0.0.1-alpha3},
  month        = jul,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.1a5},
  doi          = {10.5281/zenodo.13135902},
  url          = {https://doi.org/10.5281/zenodo.13135902}
}