# What is Helical ?

Helical provides a framework for state-of-the-art pre-trained bio foundation models on genomics and transcriptomics modalities.

Helical simplifies the entire application lifecycle when building with bio foundation models. You will be able to:

- Leverage the latest bio foundation models through our easy-to-use python package
- Run example notebooks on key downstream tasks from examples

We will update this repo on a regular basis with new models, benchmarks, modalities and functions - so stay tuned.
Let’s build the most exciting AI-for-Bio community together!

## What's new?
### New Larger Geneformer Models
We have integrated the new Geneformer models which are larger and have been trained on more data. Find out which models have been integrated into the Geneformer suite in the [model card](./model_cards/geneformer.md). Check out the our notebook on drug perturbation prediction using different Geneformer scalings [here](./notebooks/Geneformer-Series-Comparison.ipynb).

### TranscriptFormer
We have integrated [TranscriptFormer](https://github.com/czi-ai/transcriptformer) into our helical package and have made a model card for it in our [Transcriptformer model folder](./model_cards/transcriptformer.md). If you would like to test the model, take a look at our [example notebook](./notebooks/Geneformer-vs-TranscriptFormer.ipynb)!

### 🧬 Introducing Helix-mRNA-v0: Unlocking new frontiers & use cases in mRNA therapy 🧬
We’re thrilled to announce the release of our first-ever mRNA Bio Foundation Model, designed to:

1) Be Efficient, handling long sequence lengths effortlessly

2) Balance Diversity & Specificity, leveraging a 2-step pre-training approach

3) Deliver High-Resolution, using single nucleotides as a resolution

Check out our <a href="https://www.helical-ai.com/blog/helix-mrna-v0" target="_blank">blog post</a> to learn more about our approach and read the <a href="https://helical.readthedocs.io/en/latest/model_cards/helix_mrna/" target="_blank">model card</a> to get started.

## Installation

We recommend installing Helical within a conda environment with the commands below (run them in your terminal) - this step is optional:
```bash
conda create --name helical-package python=3.11.8
conda activate helical-package
```

To install the latest pip release of our Helical package, you can run the command below:
```bash
pip install helical
```

To install the latest Helical package, you can run the command below:
```bash
pip install --upgrade git+https://github.com/helicalAI/helical.git
```

Alternatively, clone the repo and install it:
```bash
git clone https://github.com/helicalAI/helical.git
pip install .
```

[Optional] To install mamba-ssm and causal-conv1d use the command below:
```bash
pip install helical[mamba-ssm]
```
or in case you're installing from the Helical repo cloned locally:
```bash
pip install .[mamba-ssm]
```

Note: make sure your machine has GPU(s) and Cuda installed. Currently this is a requirement for the packages mamba-ssm and causal-conv1d.

### Singularity (Optional)
If you desire to run your code in a singularity file, you can use the <a href="https://github.com/helicalAI/helical/blob/release/singularity.def" target="_blank">singularity.def</a> file and build an apptainer with it:
```
apptainer build --sandbox singularity/helical singularity.def
```

and then shell into the sandbox container (use the --nv flag if you have a GPU available):
```
apptainer shell --nv --fakeroot singularity/helical/
```

### RNA models:
- [Helix-mRNA](./model_cards/helix_mrna.md)
- [Mamba2-mRNA](./model_cards/mamba2_mrna.md)
- [Geneformer](./model_cards/geneformer.md)
- [scGPT](./model_cards/scgpt.md)
- [Universal Cell Embedding (UCE)](./model_cards/uce.md)
- [TranscriptFormer](./model_cards/transcriptformer.md)

### DNA models:
- [HyenaDNA](./model_cards/hyenadna.md)
- [Caduceus](./model_cards/caduceus.md)
- [Evo 2](./model_cards/evo_2.md)


## Demo & Use Cases

To run examples, be sure to have installed the Helical package (see Installation) and that it is up-to-date.

You can look directly into the example folder above and download the script of your choice, look into our [documentation](https://helical.readthedocs.io/) for step-by-step guides or directly clone the repository using:
```
git clone https://github.com/helicalAI/helical.git
```
Within the `example/notebooks` folder, open the notebook of your choice. We recommend starting with `Quick-Start-Tutorial.ipynb`.

### Current Examples:

| Example | Description | Colab |
| ----------- | ----------- |----------- |                                                        
|[Quick-Start-Tutorial.ipynb](./notebooks/Quick-Start-Tutorial.ipynb)| A tutorial to quickly get used to the helical package and environment. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Quick-Start-Tutorial.ipynb)|
|[Helix-mRNA.ipynb](./notebooks/Helix-mRNA.ipynb)|An example of how to use the Helix-mRNA model.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Helix-mRNA.ipynb) |
|[Geneformer-vs-TranscriptFormer.ipynb](./notebooks/Geneformer-vs-TranscriptFormer.ipynb) | Zero-Shot Reference Mapping with Geneformer & TranscriptFormer and compare the outcomes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Geneformer-vs-TranscriptFormer.ipynb) |
|[Hyena-DNA-Inference.ipynb](./notebooks/Hyena-DNA-Inference.ipynb)|An example how to do probing with HyenaDNA by training a neural network on 18 downstream classification tasks.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Hyena-DNA-Inference.ipynb)|
|[Cell-Type-Annotation.ipynb](./notebooks/Cell-Type-Annotation.ipynb)|An example how to do probing with scGPT by training a neural network to predict cell type annotations.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Cell-Type-Annotation.ipynb) |
|[Cell-Type-Classification-Fine-Tuning.ipynb](./notebooks/Cell-Type-Classification-Fine-Tuning.ipynb)|An example how to fine-tune different models on classification tasks.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Cell-Type-Classification-Fine-Tuning.ipynb) |
|[HyenaDNA-Fine-Tuning.ipynb](./notebooks/HyenaDNA-Fine-Tuning.ipynb)|An example of how to fine-tune the HyenaDNA model on downstream benchmarks.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/HyenaDNA-Fine-Tuning.ipynb) |
|[Cell-Gene-Cls-embedding-generation.ipynb](./examples/notebooks/Cell-Gene-Cls-embedding-generation.ipynb)|A notebook explaining the different embedding modes of single cell RNA models.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Cell-Gene-Cls-embedding-generation.ipynb) |
|[Geneformer-Series-Comparison.ipynb](./notebooks/Geneformer-Series-Comparison.ipynb)|A zero shot comparison between Geneformer model scaling on drug perturbation prediction|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Geneformer-Series-Comparison.ipynb) |

## Stuck somewhere ? Other ideas ?
We are eager to help you and interact with you:


- Join our [Slack channel](https://dk1sxv04.eu1.hubspotlinksfree.com/Ctc/L2+113/dk1sxv04/VWtlqj8M7nFNVf1vhw52bPfMW8wLjj95ptQw7N1k24YY3m2ndW8wLKSR6lZ3ldW7fZmPx5PxJ2lW8mYJtq5xWH5BVsxw821cWpdKW8CYXdj753XHSW8b5vG-7PTQ2LW1zs6x622rZxDW6930hX7RPKh3N5-trBXyRHkwVfJ3Zs3wRQV_N5NbYL3-lm47W1HvYX63pJp9cW6QXY-x6QsWMTW8G5jZh7T4vphN4Qtr7dMCxlJW8rM1-Y42pS-PW5sfJbh4FyRMhW5mHPkD4yCl56W36YW1_4GpPrGW7-sRYG1gXy8hMXqK6Sp5p69W8YTpvd3tC80SW2PTYtr6hP0dxW863B5F4KNCYkVFSWl390bSlQW78rxWn7JbS3LW14ZJ735n7SpFVSVlQr7lm7vwVlWslf6g9JRQf8mBL3b04) where you can discuss applications of bio foundation models.
- You can also open Github issues [here](https://github.com/helicalAI/helical/issues).

## Why should I use Helical & what to expect in the future?
If you are (or plan to) working with bio foundation models s.a. Geneformer or UCE on RNA and DNA data, Helical will be your best buddy! We provide and improve on:


- Up-to-date model library
- A unified API for all models
- User-facing abstractions tailored to computational biologists, researchers & AI developers
- Innovative use case and application examples and ideas
- Efficient data processing & code-base

We will continuously upload the latest model, publish benchmarks and make our code more efficient.


## Acknowledgements

A lot of our models have been published by talented authors developing these exciting technologies. We sincerely thank the authors of the following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT/)
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [UCE](https://github.com/snap-stanford/UCE)
- [TranscriptFormer](https://github.com/czi-ai/transcriptformer)
- [HyenaDNA](https://github.com/HazyResearch/hyena-dna)
- [anndata](https://github.com/scverse/anndata)
- [scanpy](https://github.com/scverse/scanpy)
- [transformers](https://github.com/huggingface/transformers)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [GenePT](https://github.com/yiqunchen/GenePT)
- [Caduceus](https://github.com/kuleshov-group/caduceus)
- [Evo2](https://github.com/ArcInstitute/evo2)

### Licenses

You can find the Licenses for each model implementation in the model repositories:

- [Helix-mRNA](https://github.com/helicalAI/helical/blob/release/helical/models/helix_mrna/LICENSE)
- [Mamba2-mRNA](https://github.com/helicalAI/helical/blob/release/helical/models/mamba2_mrna/LICENSE)
- [scGPT](https://github.com/helicalAI/helical/blob/release/helical/models/scgpt/LICENSE)
- [Geneformer](https://github.com/helicalAI/helical/blob/release/helical/models/geneformer/LICENSE)
- [UCE](https://github.com/helicalAI/helical/blob/release/helical/models/uce/LICENSE)
- [TranscriptFormer](https://github.com/helicalAI/helical/blob/release/helical/models/transcriptformer/LICENSE)
- [HyenaDNA](https://github.com/helicalAI/helical/blob/release/helical/models/hyena_dna/LICENSE)
- [Evo2](https://github.com/helicalAI/helical/blob/release/helical/models/evo_2/LICENSE)

## Citation

Please use this BibTeX to cite this repository in your publications:

```bibtex
@software{allard_2024_13135902,
  author       = {Helical Team},
  title        = {helicalAI/helical: v1.1.0},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.1.0},
  doi          = {10.5281/zenodo.13135902},
  url          = {https://doi.org/10.5281/zenodo.13135902}
}
```