from setuptools import find_packages, setup
from pathlib import Path

setup(
    name='parameter-space',
    version='0.1.0',
    license="CC BY 4.0",
    author='Tom Bland',
    author_email='tom_bland@hotmail.co.uk',
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'pandas',
                      'pde-rk'],
    description='Tools for performing model parameter space analysis'
)
