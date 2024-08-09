from setuptools import setup, find_packages
# from helical.version import __version__

setup(
    name='helical',
    # version=__version__,

    url='https://github.com/helicalAI/helical-package.git',
    author='Benoit Putzeys, Maxime Allard',
    author_email='benoit@helical-ai.com, maxime@helical-ai.com',
    packages=find_packages(),
    install_requires=[
        'requests==2.32.0',
        'pandas==2.2.2',
        'anndata==0.10.7',
        'numpy==1.26.4',
        'scikit-learn>=1.2.2',
        'scipy==1.13.1',
        'gitpython==3.1.43',
        'torch==2.3.0',
        'torchvision==0.18.0',
        'accelerate==0.29.3',
        'transformers==4.26.1',
        'loompy==3.0.7',
        'scib==1.1.5',
        'scikit-misc==0.3.1',
        'torchtext==0.18.0',
        'azure-identity==1.16.0',
        'azure-storage-blob==12.19.1',
        'azure-core==1.30.1',
        'einops==0.8.0',
        'omegaconf==2.3.0',
        'hydra-core==1.3.2',
        'tensorflow==2.17.0',
        'louvain==0.8.2',
        'pyensembl==2.3.13',
        'datasets==2.20.0',
    ],  
)
