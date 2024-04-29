from setuptools import setup, find_packages

from helical.version import __version__

setup(
    name='helical',
    version=__version__,

    url='https://github.com/helicalAI/helical-package.git',
    author='Benoit Putzeys, Maxime Allard',
    author_email='benoit@helical-ai.com, maxime@helical-ai.com',
    packages=find_packages(),
    data_files=[('helical/models/uce/', ['helical/models/uce/args.json']), 
                ('helical/models/scgpt/', ['helical/models/uce/args.json']),
                ('helical/models/geneformer/', ['helical/models/uce/args.json'])
                ],
    install_requires=[
        'requests==2.31.0',
        'pandas==2.2.2',
        'anndata==0.10.7',
        'numpy==1.26.4',
        'scikit-learn==1.2.2',
        'gitpython==3.1.43',
        'torch==2.0.0',
        'accelerate==0.29.3',
        'transformers==4.35.0',
        'loompy==3.0.7',
        'scib==1.1.5',
        'datasets==2.14.7',
        'ipython==8.24.0',
        'torchtext==0.15.1',
        'ipykernel==6.29.3'
    ],  
)