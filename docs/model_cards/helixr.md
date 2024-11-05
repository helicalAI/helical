# Model Card for Geneformer

## Model Details

**Model Name:** HelixR  
**Model Versions:** 1.0
**Model Description:** RNAmamba is a state-space model leveraging the Mamba2 architecture, trained on a comprehensive corpus of RNA sequences at single-nucleotide resolution. Unlike traditional transformer-based approaches, it utilizes selective state spaces and structured state space sequences (S4) to efficiently process long RNA sequences while capturing both local nucleotide patterns and long-range dependencies. The model's selective scan mechanism allows it to analyze RNA sequences up to hundreds of thousands of nucleotides in length, making it particularly suitable for studying full-length transcripts, splice variants, and complex RNA structural elements. Its linear time complexity with respect to sequence length (O(n)) makes it significantly more efficient than quadratic-complexity attention-based models when processing long RNA sequences.

**Example Usage**
```python
from helical import HelixR, HelixRConfig

helixr = HelixR(HelixRConfig(batch_size=5, device='cuda'))

dataset = helixr.process_data(rna_sequence_strings_list)

embeddings = helixr.get_embeddings(dataset)
print("HelixR embeddings shape: ", embeddings.shape)
```

## How To Fine-Tune

```python
from helical import HelixRFineTuningModel, HelixRConfig

helixr_config = HelixRConfig(batch_size=5, device='cuda')
helixr_fine_tune = HelixRFineTuningModel(helixr_config=helixr_config, fine_tuning_head='regression', output_size=1)

train_dataset = helixr_fine_tune.process_data(rna_sequence_strings_list)

helixr_fine_tune.train(train_dataset=train_dataset, train_labels=output_labels)

outputs = helixr_fine_tune.get_outputs(eval_dataset)

print("HelixR fine-tuned model output shape", outputs.shape)
```

