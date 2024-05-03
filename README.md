# Helical Package

![Logo](https://github.com/helicalAI/helical-package/blob/main/assets/logo1.png)

## Prerequisites

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
