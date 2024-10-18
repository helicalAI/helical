# Model Card for HyenaDNA

## Model Details

**Model Name:** HyenaDNA  \
**Model Version:** 1.0  \
**Model Description:** HyenaDNA, based on the Hyena architecture, is designed for long-range genomic sequence analysis with single nucleotide resolution. 

## Model Developers

**Developed By:** Eric Nguyen, Michael Poli, Marjan Faizi, Armin W. Thomas, Callum Birch Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen A. Baccus, Christopher Ré \
**Institutions:** Stanford University, Harvard University, SynTensor, Mila, Université de Montréal  \
**Contact Information:** [GitHub Repository](https://github.com/HazyResearch/hyena-dna)  \
**License:** Apache 2.0 

## Model Type

**Architecture:** Decoder-only sequence-to-sequence  \
**Domain:** Genomics  \
**Input Data:** DNA sequences at single nucleotide resolution 

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
- We create the probing results with the pre-trained HyenaDNA model and compare it to the results from the paper. We provide the notebook to re-produce our results.
- The tutorial [Hyena-DNA-Inference.ipynb](https://helical.readthedocs.io/en/latest/examples/notebooks/Hyena-DNA-Inference.html) was used as a basis to create this comparison, as well as the values from the [Hyena](https://arxiv.org/pdf/2306.15794) and the [Nucleotide transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1.full.pdf) (NT) papers.
- Probing was used for the 18 downstream tasks, where the HyenaDNA embeddings of nucleotide sequences were used as features to a simpler neural network.
- The same neural network with the same hyperparameters across all the tasks was used to generate these results.
- Our results underperform in comparison to the fine-tuned models. This is due to the much larger models being used for the NT, while the Hyena model was pre-trained from scratch for the better performances. 


|Dataset       |Metric       |HyenaDNA pre-trained (probing) - Helical     |NT (fine-tuned) - Original          |GPT - Original        |HyenaDNA pretrained (fine-tuned) - Original|HyenaDNA not pretrained - Original|
|    :----:    |    :----:   |    :----:   |    :----:   |    :----:   |    :----:   |    :----:   |
|H4ac|MCC|33.27%|50.10%|36.40%|**63.70%**|43.50%|
|H3K36me3|MCC|46.65%|63.20%|47.80%|**65.30%**|53.40%|
|splice_sites_donors|F1|77.08%|**98.40%**|98.10%|97.30%|96.50%|
|splice_sites_acceptors|F1|77.20%|**99.00%**|97.60%|96.60%|96.60|%
|H3|MCC|72.07%|81.40%|75.80%|**81.70%**|79.90%|
|H4|MCC|72.35%|**82.20%**|77.70%|79.60%|79.10%|
|H3K4me3|MCC|24.04%|42.10%|28.30%|**61.20%**|40.20%|
|splice_sites_all|F1|57.15%|**98.30%**|98.00%|97.90%|97.30%|
|H3K4me1|MCC|38.11%|55.90%|38.70%|**57.10%**|43.40%|
|H3K14ac|MCC|36.69%|55.00%|41.60%|**66.30%**|48.00%|
|enhancers_types|MCC|34.62%|47.40%|51.90%|**55.70%**| 48.40%|
|promoter_no_tata|F1|93.84%|**97.70%**|96.60%|96.60%|96.50%|
|H3K79me3|MCC|54.54%|64.20%|58.90%|**71.60%**|59.70%|
|H3K4me2|MCC|27.00%|32.60%|28.80%|**53.90%**|34.50%|
|promoter_tata|F1|91.91%|96.40%|96.60%|**96.70%**|96.10%|
|enhancers|MCC|48.02%|58.00%|59.30%|**62.60%**|58.60%|
|H3K9ac|MCC|43.01%|57.50%|49.20%|**65.10%**|52.60%|
|promoter_all|F1|93.99%|**97.40%**|96.30%|96.50%|96.10%|

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
from helical import HyenaDNA, HyenaDNAConfig

hyena_config = HyenaDNAConfig(model_name = "hyenadna-tiny-1k-seqlen-d256")
model = HyenaDNA(configurer = hyena_config)   
sequence = 'ACTG' * 1024
tokenized_sequence = model.process_data(sequence)
embeddings = model.get_embeddings(tokenized_sequence)

print(embeddings.shape)
```

## How to Fine-Tune
```python
from datasets import load_dataset
from helical import HyenaDNAConfig, HyenaDNAFineTuningModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load a Hugging Face dataset and task type
ds = load_dataset("dataset", "task")

# Define the desired configs
config = HyenaDNAConfig(device=device, batch_size=10)

# Define the fine-tuning model with the configs we instantiated above
hyena_fine_tune = HyenaDNAFineTuningModel(config, "classification", number_unique_outputs)

# Prepare the sequences for input to the model
input_dataset = hyena_fine_tune.process_data(ds["train"]["sequence"])

# train the fine-tuning model on some downstream task
hyena_fine_tune.train(input_dataset, ds["train"]["label"])

```

## Citation

@article{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faizi, Marjan and others},
  journal={arXiv preprint arXiv:2306.15794},
  year={2023}
}

When using HyenaDNA, please cite the original paper and use the DOI link provided in the GitHub repository.
