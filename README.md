<div align="center">
  <p><a href="https://helical.readthedocs.io/"/>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo_and_text_v2_white.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo_and_text_v2.png">
    <img alt="Helical Logo" src="docs/assets/logo_and_text_v2_white.png" width="300">
  </picture>
  </a></p>
</div>


# What is Helical ?

Helical provides a framework for state-of-the-art pre-trained bio foundation models on genomics and transcriptomics modalities.

Helical simplifies the entire application lifecycle when building with bio foundation models. You will be able to:
- Leverage the latest bio foundation models through our easy-to-use python package
- Run example notebooks on key downstream tasks from examples

We will update this repo on a regular basis with new models, benchmarks, modalities and functions - so stay tuned.
Let’s build the most exciting AI-for-Bio community together!
<div align="center">

![Workflow](https://github.com/helicalAI/helical/actions/workflows/main.yml/badge.svg) &nbsp;
![Workflow](https://github.com/helicalAI/helical/actions/workflows/github-code-scanning/codeql/badge.svg) &nbsp;
[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://helical.readthedocs.io/) &nbsp;
[![PyPI version](https://badge.fury.io/py/helical.svg)](https://badge.fury.io/py/helical) &nbsp;
![GitHub contributors](https://img.shields.io/github/contributors/helicalAI/helical) &nbsp;

</div>

## Installation

We recommend installing Helical within a conda environment with the commands below (run them in your terminal) - this step is optional:
```
conda create --name helical-package python=3.11.8
conda activate helical-package
```

To install the latest pip release of our Helical package, you can run the command below:
```
pip install helical
```

To install the latest Helical package, you can run the command below:
```
pip install --upgrade git+https://github.com/helicalAI/helical.git
```

### Singularity (Optional)
If you desire to run your code in a singularity file, you can use the [singularity.def](./singularity.def) file and build an apptainer with it:
```
apptainer build --sandbox singularity/helical singularity.def
```

and then shell into the sandbox container (use the --nv flag if you have a GPU available):
```
apptainer shell --nv --fakeroot singularity/helical/
```

## Installation
### RNA models:
- [Helix-mRNA]((https://helical.readthedocs.io/en/latest/model_cards/helix_mrna/))
- [Mamba2-mRNA]((https://helical.readthedocs.io/en/latest/model_cards/mamba2_mrna/))
- [Geneformer](https://helical.readthedocs.io/en/latest/model_cards/geneformer/)
- [scGPT](https://helical.readthedocs.io/en/latest/model_cards/scgpt/)
- [Universal Cell Embedding (UCE)](https://helical.readthedocs.io/en/latest/model_cards/uce/)

### DNA models:
- [HyenaDNA](https://helical.readthedocs.io/en/latest/model_cards/hyenadna/)


## Demo & Use Cases

To run examples, be sure to have installed the Helical package (see Installation) and that it is up-to-date.

You can look directly into the example folder above and download the script of your choice, look into our [documentation](https://helical.readthedocs.io/) for step-by-step guides or directly clone the repository using:
```
git clone https://github.com/helicalAI/helical.git
```
Within the `examples/notebooks` folder, open the notebook of your choice. We recommend starting with `Quick-Start-Tutorial.ipynb`

### Current Examples:

| Example | Description | Colab |
| ----------- | ----------- |----------- |                                                        
|[Quick-Start-Tutorial.ipynb](./examples/notebooks/Quick-Start-Tutorial.ipynb)| A tutorial to quickly get used to the helical package and environment. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Quick-Start-Tutorial.ipynb)|
|[Geneformer-vs-UCE.ipynb](./examples/notebooks/Geneformer-vs-UCE.ipynb) | Zero-Shot Reference Mapping with Geneformer & UCE and compare the outcomes. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Geneformer-vs-UCE.ipynb) |
|[Hyena-DNA-Inference.ipynb](./examples/notebooks/Hyena-DNA-Inference.ipynb)|An example how to do probing with HyenaDNA by training a neural network on 18 downstream classification tasks.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Hyena-Dna-Inference.ipynb) |
|[Cell-Type-Annotation.ipynb](./examples/notebooks/Cell-Type-Annotation.ipynb)|An example how to do probing with scGPT by training a neural network to predict cell type annotations.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Cell-Type-Annotation.ipynb) |
|[Cell-Type-Classification-Fine-Tuning.ipynb](./examples/notebooks/Cell-Type-Classification-Fine-Tuning.ipynb)|An example how to fine-tune different models on classification tasks.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/Cell-Type-Classification-Fine-Tuning.ipynb) |
|[HyenaDNA-Fine-Tuning.ipynb](./examples/notebooks/HyenaDNA-Fine-Tuning.ipynb)|An example of how to fine-tune the HyenaDNA model on downstream benchmarks.|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helicalAI/helical/blob/main/examples/notebooks/HyenaDNA-Fine-Tuning.ipynb) |
| Coming Soon | New models such as SCimilarity, scVI; benchmarking scripts; new use cases; others |

## Stuck somewhere ? Other ideas ?
We are eager to help you and interact with you. Reach out via support@helical-ai.com. 
You can also open github issues here.

## Why should I use Helical & what to expect in the future?
If you are (or plan to) working with bio foundation models s.a. Geneformer or UCE on RNA and DNA data, Helical will be your best buddy! We provide and improve on:
- Up-to-date model library
- A unified API for all models
- User-facing abstractions tailored to computational biologists, researchers & AI developers
- Innovative use case and application examples and ideas
- Efficient data processing & code-base

We will continuously upload the latest model, publish benchmarks and make our code more efficient.


## Acknowledgements

A lot of our models have been published by talend authors developing these exciting technologies. We sincerely thank the authors of the following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT/)
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [UCE](https://github.com/snap-stanford/UCE)
- [HyenaDNA](https://github.com/HazyResearch/hyena-dna)
- [anndata](https://github.com/scverse/anndata)
- [scanpy](https://github.com/scverse/scanpy)
- [transformers](https://github.com/huggingface/transformers)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)

### Licenses

You can find the Licenses for each model implementation in the model repositories:

- [Helix-mRNA](https://github.com/helicalAI/helical/blob/release/helical/models/helix_mrna/LICENSE)
- [Mamba2-mRNA](https://github.com/helicalAI/helical/blob/release/helical/models/mamba2_mrna/LICENSE)
- [scGPT](https://github.com/helicalAI/helical/blob/release/helical/models/scgpt/LICENSE)
- [Geneformer](https://github.com/helicalAI/helical/blob/release/helical/models/geneformer/LICENSE)
- [UCE](https://github.com/helicalAI/helical/blob/release/helical/models/uce/LICENSE)
- [HyenaDNA](https://github.com/helicalAI/helical/blob/release/helical/models/hyena_dna/LICENSE)

## Citation

Please use this BibTeX to cite this repository in your publications:

```bibtex
@software{allard_2024_13135902,
  author       = {Helical Team},
  title        = {helicalAI/helical: v0.0.1-alpha6},
  month        = nov,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.0.1a6},
  doi          = {10.5281/zenodo.13135902},
  url          = {https://doi.org/10.5281/zenodo.13135902}
}
```

