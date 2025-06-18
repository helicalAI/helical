# Model Card for Evo 2

## Model Details

**Model Name:** Evo 2  

**Model Versions:** 1B, 7B and 40B

**Model Description:** Evo 2 is a next-generation genomic model that integrates DNA, RNA, and protein data across all domains of life. It leverages the StripedHyena 2 architecture, combining convolutional, linear attention, and state-space models to efficiently process long sequences and capture complex biological patterns. Evo 2 is trained on a vast dataset encompassing trillions of nucleotides from eukaryotic and prokaryotic genomes, enabling broad cross-species applications and insights into human diseases, agriculture, and environmental science.

## Model Developers

Arc Institute, Stanford University, NVIDIA, Liquid AI, University of California, Berkeley, Goodfire, Columbia University

**Contact Information:** 

[Patrick D. Hsu](mailto:patrick@arcinstitute.org)

[Brian L. Hie (Stanford Email)](mailto:brianhie@stanford.edu)

[Brian L. Hie (Arc Institute Email)](mailto:brian.hie@arcinstitute.org)

**License:** 
Apache-2.0 

## Model Type

**Architecture:** StripedHyena 2 (Multi-hybrid)  
**Domain:** Genomics and Proteomics  
**Input Data:** DNA, RNA, and protein sequences.

## Model Purpose

**Intended Use:**  
- **Genomic Analysis:** Predicting mutation impacts, annotating genomes, and identifying essential genes.
- **Protein Analysis:** Understanding protein structure and function.
- **Cross-Species Applications:** Facilitating insights across different domains of life.
- **Biological Design:** Generating complex biological systems.

**Use Cases**

- **Variant Impact Prediction:** Accurately predicting the effects of mutations across species.
- **Gene Essentiality Analysis:** Identifying crucial genes in various organisms.
- **Biological Design:** Designing genome-scale sequences and controlling chromatin accessibility.
- **Therapeutic Applications:** Informing human disease research and agricultural innovations.

## Training Data

**Pretraining:**  

- 1B model is trained on 1T tokens up to 8,192 tokens
- 7B base model is trained on 2.1T tokens up to 8,192 tokens
- 7B model is trained on 4T tokens up to 1M tokens
- 40B base model is trained on 6.6T tokens at length 1,024 tokens and 1.1T tokens at length 8,192 tokens
- 40B model is trained rained on 9.3T tokens up to 1M tokens
- For both the 7B and 40B models, a multi-stage pretraining approach was implemented. This began with an initial pretraining phase focused exclusively on sequences of 8,192 (1024 as well as 8,192 for the 40B model) tokens, followed by a progressive increase in context length up to 1M tokens.

**Preprocessing:**  

- Utilizes novel data augmentation and weighting strategies to prioritize functional genetic elements.

## Model Performance

**Evaluation Metrics:**  
- Accuracy in mutation impact prediction, gene essentiality identification, and genome annotation tasks.
- AUROC for exon classification tasks.

## Model Limitations

**Known Limitations:**  
- Performance may vary based on the complexity and length of input sequences.
- Limited by the availability of diverse and high-quality training data.
- Only runnable on NVIDIA GPUs with compute capability ≥8.9
- These are very large models and need significant compute to run

## Install the package

### Via the Docker image

```bash
git clone https://github.com/helicalAI/helical.git

cd helical/helical/models/evo_2

docker build -t helical_with_evo_2 .

docker run -it --gpus all helical_with_evo_2
```

### Installing within a conda environment

```bash
conda create -n helical-env-with-evo-2 python=3.11
conda activate helical-env-with-evo-2

conda install cuda-toolkit=12.4 -c nvidia

export CUDNN_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include

pip install torch==2.6.0
pip install "helical[evo-2]@git+https://github.com/helicalAI/helical.git"

git clone https://github.com/Zymrael/vortex.git
cd vortex
git checkout f243e8e
sed -i 's/torch==2.5.1/torch==2.6.0/g' pyproject.toml
make setup-full
cd ..

pip install torch==2.6.0 torchvision
```

## How to Use

**Input Format**  
- DNA, RNA, and protein sequences in standard formats.

**Output Format**  
- Sequence embeddings and predictions for various biological tasks.

**Example Usage For Getting Embeddings**

```python 
from helical.models.evo_2 import Evo2, Evo2Config

evo2_config = Evo2Config(batch_size=1)

evo2 = Evo2(configurer=evo2_config)

sequences = ["ACGT" * 1000]

dataset = evo2.process_data(data)

embeddings = evo2.get_embeddings(dataset)
# Get the last embedding of each sequence
print(embeddings["embeddings"][0][embeddings["original_lengths"][0]-1])
print(embeddings["embeddings"][1][embeddings["original_lengths"][1]-1])
print(embeddings["original_lengths"])
```

**Example Usage For Sequence Generation**

```python
from helical.models.evo_2 import Evo2, Evo2Config

evo2_config = Evo2Config(batch_size=1)

evo2 = Evo2(configurer=evo2_config)

sequences = ["ACGT" * 1000]

dataset = evo2.process_data(data)

generate = evo2.generate(dataset)

# Print the generated sequences
print(generate)
```

## Citation
```bibtex
@article {Brixi2025.02.18.638918,
	author = {Brixi, Garyk and Durrant, Matthew G and Ku, Jerome and Poli, Michael and Brockman, Greg and Chang, Daniel and Gonzalez, Gabriel A and King, Samuel H and Li, David B and Merchant, Aditi T and Naghipourfar, Mohsen and Nguyen, Eric and Ricci-Tam, Chiara and Romero, David W and Sun, Gwanggyu and Taghibakshi, Ali and Vorontsov, Anton and Yang, Brandon and Deng, Myra and Gorton, Liv and Nguyen, Nam and Wang, Nicholas K and Adams, Etowah and Baccus, Stephen A and Dillmann, Steven and Ermon, Stefano and Guo, Daniel and Ilango, Rajesh and Janik, Ken and Lu, Amy X and Mehta, Reshma and Mofrad, Mohammad R.K. and Ng, Madelena Y and Pannu, Jaspreet and Re, Christopher and Schmok, Jonathan C and St. John, John and Sullivan, Jeremy and Zhu, Kevin and Zynda, Greg and Balsam, Daniel and Collison, Patrick and Costa, Anthony B. and Hernandez-Boussard, Tina and Ho, Eric and Liu, Ming-Yu and McGrath, Tom and Powell, Kimberly and Burke, Dave P. and Goodarzi, Hani and Hsu, Patrick D and Hie, Brian},
	title = {Genome modeling and design across all domains of life with Evo 2},
	elocation-id = {2025.02.18.638918},
	year = {2025},
	doi = {10.1101/2025.02.18.638918},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/02/21/2025.02.18.638918},
	eprint = {https://www.biorxiv.org/content/early/2025/02/21/2025.02.18.638918.full.pdf},
	journal = {bioRxiv}
}
```

## Contact
[Helical Support](mailto:support@helical-ai.com)
