# parameter-space-2d

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

The repository contains a couple of notebooks with instructions for performing analysis, using the Goehring et al. (2011) PAR polarity model as an example. To run the notebooks interactively, click here:

<a target="_blank" href="https://colab.research.google.com/github/tsmbland/parameter-space-2d/blob/master/scripts/INDEX.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height=20/></a>



## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

