<div align="center">
  <img src="https://github.com/helicalAI/helical-package/blob/main/assets/logo%2Bname.png" alt="Logo" width="304" height="110">
</div>

## What is Helical ?

Helical provides a framework for and gathers state-of-the-art pre-trained bio foundation models on genomics and transcriptomics modalities.

Helical simplifies the entire application lifecycle when building with bio foundation models. You will be able to:
Leverage the latest bio foundation models through our easy-to-use python package
Run example notebooks on key downstream tasks from examples

Helical is backed by the main deep learning and bio libraries, such as [xxx].

We will update this repo on a bi-weekly basis with new models, benchmarks, modalities and functions - so stay tuned.
Let’s build the most exciting AI-for-Bio community together !

## Installation

We recommend installing Helical within a conda environment with the commands below (run them in your terminal):
```
conda create --name helical-package python=3.11.8
conda activate helical-package
```
```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/helicalAI/helical-package.git
```

This repository is tested on Python xxx+, Flax xxx+,...2.6+.















We recommend using a conda environment with `Pyhton 3.11.8`.
```
conda create --name helical-package python=3.11.8
conda activate helical-package
```
Install the `helical-package`:

```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/helicalAI/helical-package.git
```
You can then run the files in the `examples` folder.


## Documentation
To start the docs please install the requirements:

```
cd docs
pip install -r requirements.txt
```

Then go back into the root folder and compile the code:

```
SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -f -d 0 -o docs/source ./
jupyter-book build ./   

```
And then you should be able to open the docs in ./build/html/index.html .
