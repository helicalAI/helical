from setuptools import setup, find_packages
from helical.version import __version__

setup(
    name='helical',
    version=__version__,

    url='https://github.com/helicalAI/helical-package.git',
    author='Benoit Putzeys, Maxime Allard',
    author_email='benoit@helical-ai.com, maxime@helical-ai.com',
    packages=find_packages(),
    install_requires=[
        'requests==2.31.0',
        'pandas==2.2.2',
        'anndata==0.10.7',
        'numpy==1.26.4',
        'scikit-learn==1.2.2',
        'gitpython==3.1.43',
        'torch==2.3.0',
        'accelerate==0.29.3',
        'transformers==4.26.1',
        'loompy==3.0.7',
        'scib==1.1.5',
        'datasets==2.14.7',
        'ipython==8.24.0',
        'torchtext==0.18.0',
        'ipykernel==6.29.3',
        'IProgress==0.4',
        "ipywidgets==8.1.2",
        'azure-identity==1.16.0',
        'azure-storage-blob==12.19.1',
        'azure-core==1.30.1',
        'einops==0.8.0',
        'omegaconf==2.3.0',
    ],  
)