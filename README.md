# parameter-space-2d

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tsmbland/parameter-space/HEAD?filepath=%2Fscripts/INDEX.ipynb)
[![CC BY 4.0][cc-by-shield]][cc-by]
[![PyPi version](https://badgen.net/pypi/v/parameter-space-2d/)](https://pypi.org/project/parameter-space-2d)

<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/parameter-space-2d/master/docs/example_maps.png" width="100%" height="100%"/>
</p>

Tools for performing 2D parameter space analysis for deterministic models

## Method

Parameter space maps are built up by performing an iterative grid search, preferentially exploring parameter regimes 
close to detected boundaries. Various checks are in place to ensure that any detected boundaries are fully explored.


<p align="center">
    <img src="https://raw.githubusercontent.com/tsmbland/parameter-space-2d/master/docs/method.png" width="100%" height="100%"/>
</p>

_Example of a model with two states (blue and orange) and variable input parameters β and ε_


## Installation

    pip install parameter-space-2d

## Instructions

The repository contains a couple of [notebooks](scripts/INDEX.ipynb) with instructions for performing analysis, using the Goehring et al. (2011) PAR polarity model as an example. To run the notebooks interactively you have a few options:

#### Option 1: Binder

To run in the cloud using Binder, click here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tsmbland/parameter-space/HEAD?filepath=%2Fscripts/INDEX.ipynb)

(Please note that it may take several minutes to open the notebook)

#### Option 2: Docker

Step 1: Open [Docker](https://www.docker.com/products/docker-desktop/) and pull the docker image (copy and paste into the terminal)

    docker pull tsmbland/parameter-space

Step 2: Run the docker container (copy and paste into the terminal)

    docker run -p 8888:8888 tsmbland/parameter-space

This will print a URL for you to copy and paste into your web browser to open up Jupyter

Step 3: When finished, delete the container and image
    
    docker container prune -f
    docker image rm tsmbland/parameter-space

#### Option 3: Conda

You can use the environment.yml file to set up a [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment on your machine from which the notebook can be run

    conda env create -f environment.yml
    conda activate parameter-space-2d
    jupyter notebook


## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

