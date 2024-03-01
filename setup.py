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
        'requests',
        'pandas',
        'anndata',
        'tiledbsoma',
        'numpy'
    ],  
)