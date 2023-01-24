from setuptools import find_packages, setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='parameter-space-2d',
    version='0.1.1',
    license="CC BY 4.0",
    author='Tom Bland',
    author_email='tom_bland@hotmail.co.uk',
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib'],
    description='Tools for performing 2D parameter space analysis for deterministic models',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
