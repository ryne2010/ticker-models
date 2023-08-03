# ! Adaped from Jupyter Dockerstack recommended for Transformers (https://github.com/ToluClassics/transformers_notebook/blob/main/Dockerfile)
# FROM jupyter/base-notebook
# ! Adaped from Jupyter Dockerstack recommended for GPU-accellerated Transformers (https://github.com/b-data/jupyterlab-python-docker-stack/blob/main/CUDA.md)
# FROM glcr.b-data.ch/jupyterlab/cuda/python/scipy
# FROM python:3.11-slim-bookworm
# FROM continuumio/miniconda3
FROM jupyter/base-notebook:python-3.10.4

# * Init & Config
ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV TOKENIZERS_PARALLELISM=true

# # Allows "conda create" to restart shell
# SHELL ["bash", "-lc"]

USER root

RUN python -m pip install --no-cache-dir --upgrade pip conda
# RUN conda init
RUN conda config --append channels conda-forge

# * Install Basic Packages
# # Prefer conda
RUN conda install -c conda-forge jupyterlab
RUN conda install -c anaconda ipykernel
RUN conda install pytorch torchvision -c pytorch
RUN conda install -c conda-forge pytorch-lightning
RUN conda install \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    tqdm \
    transformers \
    datasets\
    nltk \
    gradio \
    sentencepiece \
    seqeval
    # mwclient \
    # yfinance \
    # xgboost

# * Create PyTorch env
# RUN conda create -n "torchEnv" python=3.10 ipython
# RUN conda activate "torchEnv"
# RUN python -m ipykernel install --user --name=torchEnv

# * Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
# RUN python3 -m pip install --no-cache-dir --requirement /tmp/requirements.txt
# RUN conda install -y -q --name torchEnv -c conda-forge --file /tmp/requirements.txt
RUN conda install -y -q -c conda-forge --file /tmp/requirements.txt

# * Cleanup
# USER ${NB_UID}
USER $NB_USER
WORKDIR "${HOME}"
