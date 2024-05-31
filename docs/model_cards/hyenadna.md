# Model Card for HyenaDNA

## Model Details

**Model Name:** HyenaDNA  \
**Model Version:** 1.0  \
**Model Description:** HyenaDNA, based on the Hyena architecture, is designed for long-range genomic sequence analysis with single nucleotide resolution. \

## Model Developers

**Developed By:** Eric Nguyen, Michael Poli, Marjan Faizi, Armin W. Thomas, Callum Birch Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen A. Baccus, Christopher Ré \
**Institutions:** Stanford University, Harvard University, SynTensor, Mila, Université de Montréal  \
**Contact Information:** [GitHub Repository](https://github.com/HazyResearch/hyena-dna)  \
**License:** Apache 2.0 \

## Model Type

**Architecture:** Decoder-only sequence-to-sequence  \
**Domain:** Genomics  \
**Input Data:** DNA sequences at single nucleotide resolution \

## Model Purpose

**Intended Use:**  
- Research in genomics
- Computational biology

**Out-of-Scope Use Cases:**  
- Direct clinical applications without further validation

## Training Data

**Data Sources:**  
- Human reference genome  
- [Data Source Link](https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.26/)

## Model Performance

**Evaluation Metrics:**  
- Accuracy, Precision, Recall, F1-Score

**Performance Benchmarks:**  
- Achieves state-of-the-art on various genomic tasks

## Ethical Considerations

**Bias and Fairness:**  
- Trained only on the human reference genome

**Privacy:**  
- Uses publicly available genomic data

**Mitigations:**  
- Continuous monitoring for biases

## Model Limitations

**Known Limitations:**  
- Limited to genomic data

**Future Improvements:**  
- Expansion to include diverse genomic datasets

## How to Use

**Input Format:**  
- DNA sequence strings

**Output Format:**  
- JSON objects with genomic feature predictions

**Example Usage:**
```python
from hyenadna import HyenaDNA

model = HyenaDNA()
sequence = "ATCG..."
result = model.predict(sequence)
print(result)
```

## Citation

@article{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faizi, Marjan and others},
  journal={arXiv preprint arXiv:2306.15794},
  year={2023}
}

When using HyenaDNA, please cite the original paper and use the DOI link provided in the GitHub repository.