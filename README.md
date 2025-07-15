# Machine Learning
This repository is a series of libraries, experiments around Machine Learning.

## About Machine Learning
Machine Learning is about learning useful structures from data.

In this repository we will be writing down the theory and performing experiments (perhaps also developing useful libraries)
with implementations of approaches found in literature.

## Installing Dependencies:
Reproducing the environment (on a new machine):
1. Install docker: https://docs.docker.com/desktop/setup/install/mac-install/.
0. Install Java: `brew install --cask temurin@21`
0. Install `pyenv` and `pyenv-virtualenv` on the new machine.
0. Copy your project directory (including .python-version and requirements.txt) to the new machine.
0. cd into the project directory.
0. `pyenv install` (This will read .python-version and install the correct Python version).
0. `pyenv virtualenv 3.11 <env_name>` (Creates the virtual environment)
0. `pyenv activate <env_name>`
0. `pip install -r requirements.txt` (This installs all the packages listed in requirements.txt).
0. In VSCode `Cmd + Shift + P` ->  `Python: Select interpreter`