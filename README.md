# Machine Learning
This repository is a series of libraries, experiments around Machine Learning.

## About Machine Learning
Machine Learning is about learning useful structures from data.

In this repository we will be writing down the theory and performing experiments (perhaps also developing useful libraries)
with implementations of approaches found in literature.

## Installing Dependencies:
Reproducing the environment (on a new machine):

1. Install `pyenv` and `pyenv-virtualenv` on the new machine.
2. Copy your project directory (including .python-version and requirements.txt) to the new machine.
3. cd into the project directory.
4. `pyenv install` (This will read .python-version and install the correct Python version).
5. `pyenv virtualenv 3.11 <env_name>` (Creates the virtual environment)
6. `pyenv activate <env_name>`
7. `pip install -r requirements.txt` (This installs all the packages listed in requirements.txt).
8. In VSCode `Cmd + Shift + P` ->  `Python: Select interpreter`