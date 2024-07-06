# Neuro-Symbolic AI project

This repo contains the project for the Neuro-Symbolic AI PhD course taught at PoliTo.

The code is based on the [source code](https://github.com/whitemech/suffix-prediction-pmai2024) of the paper submitted to the PMAI@ECAI2024 workshop by Elena Umili, Gabriel Paludo Licks and Fabio Patrizi.

## Dependencies

You can find a `Dockerfile` in this repository. You can either run the code on a Docker container or use the Dockerfile as a reference for the dependencies (mainly CUDA/torch and MONA) that need to be installed. You can also find an `environment.yml` file referring to a Conda virtual environment containing the Python dependencies.

## Running the code

The file `run_all.py` is the script that runs all experiments shown in the paper.
The file `run_on_real.py` is the script that runs new experiments on real datasets.
Once the scripts finish executing, you can plot the results using `plot.py` or `plot_on_real.py`.