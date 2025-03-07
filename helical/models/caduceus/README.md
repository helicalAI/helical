# Model Card for Bio-Foundation Model

## Model Details

**Model Name:** Caduceus

**Model Version:** 1.0  

**Model Description:** The Caduceus model is built upon the long-range Mamba block, extending this approach to a BiMamba component that supports bi-directionality and to a Mamba block that additionally supports reverse complementarity (RC) of DNA. This approach allows Caduceus to outperform larger models that do not utilise bi-directionality or equivariance.

#### Note
This model has dependencies which only allow it to be run on CUDA devices.

## Model Developers

**Developed By:** Kuleshov Group @ Cornell

**Contact Information:** yzs2@cornell.edu

**License:** Apache 2.0

## Model Type

**Architecture:** Mamba-based (Bi-Directional and RC-Equivariant)

**Domain:** Genomics  

**Input Data:** DNA sequences

## Model Purpose

**Use cases:**  
- Variant effect prediction (VEP) to determine if genetic mutations affect gene expression
- Recognition of evolutionary pressure effects (conservation, co-evolution)
- Prediction of long-range effects on gene expression

## Training Data

**Training split** 
34,021 segments, extended to maximum length of 1,048,576 (2^20)

**Total size:**
~35 billion tokens/base pairs

**Data source:**
Human reference genome splits from Enformer study (Avsec et al., 2021)

**Augmentation:**
 RC (reverse complement) data augmentation used for all models except Caduceus-PS, with 50% probability of applying RC operation

## Model Limitations

**Known Limitations:**  
- Model pretraining was specifically focused on human-genome related tasks and may not generalize well to other genomes.

## How to Use

**Example Usage:**
```python
from helical.models.caduceus import CaduceusConfig, Caduceus
caduceus_config = CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118", batch_size=2, pooling_strategy="mean")
caduceus = Caduceus(configurer=caduceus_config)

sequence = ['ACTG' * int(1024/4), 'TGCA' * int(1024/4)]
processed_data = caduceus.process_data(sequence)

embeddings = caduceus.get_embeddings(processed_data)
print(embeddings.shape)
```

**Example Fine-Tuning**
```python
from helical.models.caduceus import CaduceusConfig, CaduceusFineTuningModel

input_sequences = ["ACT"*20, "ATG"*20, "ATG"*20, "CTG"*20, "TTG"*20]
labels = [0, 2, 2, 0, 1]

caduceus_config = CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118", batch_size=2, pooling_strategy="mean")
caduceus_fine_tune = CaduceusFineTuningModel(caduceus_config=caduceus_config, fine_tuning_head="classification", output_size=3)

train_dataset = caduceus_fine_tune.process_data(input_sequences)

caduceus_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

outputs = caduceus_fine_tune.get_outputs(train_dataset)
print(outputs.shape)
```

## Contact

yzs2@cornell.edu

## Citation

```bibtex
@article{schiff2024caduceus,
  title={Caduceus: Bi-directional equivariant long-range dna sequence modeling},
  author={Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and Dao, Tri and Gu, Albert and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2403.03234},
  year={2024}
}
```