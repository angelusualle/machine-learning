# Machine Learning
This repository is a series of libraries, experiments around Machine Learning.

## About Machine Learning
Machine Learning is about learning useful structures from data.

In this repository we will be writing down the theory and performing experiments (perhaps also developing useful libraries)
with implementations of approaches found in literature.

## Installing Dependencies and running the code:
For reproducibility particularly around distributed algorithms we use `docker-compose` to set up the containers and environment for running the code:
1. Install docker: https://docs.docker.com/desktop/setup/install/mac-install/.
2. For a given algorithm, there's a `/cluster` folder where you can run a variant of `docker-compose up` to set up and `docker-compose down` to tear down the environment. See the notebook for the algorithm for specific instructions.
