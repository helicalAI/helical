(intro)=

# Jupyter Book on Read the Docs

This example shows a Jupyter Book project built and published on Read the Docs.
You're encouraged to use it to get inspiration and copy & paste from the files in [the source code repository][github]. In the source repository, you will also find the relevant configuration and instructions for building Jupyter Book projects on Read the Docs.

If you are using Read the Docs for the first time, have a look at the official [Read the Docs Tutorial][tutorial].
If you are using Jupyter Book for the first time, have a look at the [official Jupyter Book documentation][jb-docs].

## Why run Jupyter Book with Read the Docs?

[Read the Docs](https://readthedocs.org/) simplifies developing Jupyter Book projects by automating building, versioning, and hosting of your project for you.
You might be familiar with Read the Docs for software documentation projects, but these features are just as relevant for science.

With Read the Docs, you can improve collaboration on your Jupyter Book project with Git (GitHub, GitLab, BitBucket etc.) and then connect the Git repository to Read the Docs.
Once Read the Docs and the git repository are connected, your project will be built and published automatically every time you commit and push changes with git.
Furthermore, if you open Pull Requests, you can preview the result as rendered by Jupyter Book.

## What is in this example?

Jupyter Book has a number of built-in features.
This is a small example book to give you a feel for how book content is structured.
It shows off a few of the major file types, as well as some sample content.
It does not go in-depth into any particular topic - check out [the Jupyter Book documentation][jb-docs] for more information.

* [Examples of Markdown](/markdown)
* [Rendering a notebook Jupyter Notebook](/notebooks)
* [A notebook written in MyST Markdown](/markdown-notebooks)

We have also added some popular features for Jupyter Book that really you shouldn't miss when building your own project with Jupyter Book and Read the Docs:

* [intersphinx to link to other documentation and Jupyter Book projects](/intersphinx)
* [sphinx-examples to show examples and results side-by-side](/sphinx-examples)
* [sphinx-hoverxref to preview cross-references](/sphinx-hoverxref)
* [sphinx-proof for logic and math, to write proofs, theorems, lemmas etc.](/sphinx-proof)


## Table of Contents

Here is an automatically generated Tabel of Contents:

```{tableofcontents}
```

## What is Helical ?

Helical provides a framework for and gathers state-of-the-art pre-trained bio foundation models on genomics and transcriptomics modalities.

Helical simplifies the entire application lifecycle when building with bio foundation models. You will be able to:
- Leverage the latest bio foundation models through our easy-to-use python package
- Run example notebooks on key downstream tasks from examples

We will update this repo on a bi-weekly basis with new models, benchmarks, modalities and functions - so stay tuned.
Let’s build the most exciting AI-for-Bio community together!

## Installation

We recommend installing Helical within a conda environment with the commands below (run them in your terminal) - this step is optional:
```
conda create --name helical-package python=3.11.8
conda activate helical-package
```
To install the Helical package, you can run the command below:
```
pip install --upgrade --force-reinstall git+https://github.com/helicalAI/helical.git
```


## Demo & Use Cases

To run examples, be sure to have installed the Helical package (see Installation) and that it is up-to-date.

You can look directly into the example folder above, look into our [documentation](https://helical.readthedocs.io/) for step-by-step guides or directly clone the repository using:
```
git clone https://github.com/helicalAI/helical.git
```
Within the `example` folder, open the notebook of your choice. We recommend starting with `Geneformer-vs-UCE.ipynb`

### RNA models:
- `Geneformer-vs-UCE.ipynb`: Zero-Shot Reference Mapping with Geneformer & UCE and compare the outcomes.
- Coming soon: new models such as scGPT, SCimilarity, scVI; benchmarking scripts; new use cases; others

### DNA models:
- Coming soon: new models such as Nucleotide Transformer; others

# Stuck somewhere ? Other ideas ?
We are eager to help you and interact with you. Reach out via rick@helical-ai.com. 
You can also open github issues here.

# Why should I use Helical & what to expect in the future?
If you are (or plan to) working with bio foundation models s.a. Geneformer or UCE on RNA and DNA data, Helical will be your best buddy! We provide and improve on:
- Up-to-date model library
- A unified API for all models
- User-facing abstractions tailored to computational biologists, researchers & AI developers
- Innovative use case and application examples and ideas
- Efficient data processing & code-base

We will continuously upload the latest model, publish benchmarks and make our code more efficient.

# Acknowledgements

A lot of our models have been published by talend authors developing these exciting technologies. We sincerely thank the authors of the following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT/)
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- [UCE](https://github.com/snap-stanford/UCE)
- [anndata](https://github.com/scverse/anndata)
- [scanpy](https://github.com/scverse/scanpy)
- [transformers](https://github.com/huggingface/transformers)


# Citation

Please use this BibTeX to cite this repository in your publications:

```
@misc{helical,
  author = {Maxime Allard, Benoit Putzeys, Rick Schneider, Mathieu Klop},
  title = {Helical Python Package},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/helicalAI/helical}},
}
