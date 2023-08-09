# ! Adaped from Jupyter Dockerstack recommended for Transformers (https://github.com/ToluClassics/transformers_notebook/blob/main/Dockerfile)
# FROM jupyter/base-notebook
# ! Adaped from Jupyter Dockerstack recommended for GPU-accellerated Transformers (https://github.com/b-data/jupyterlab-python-docker-stack/blob/main/CUDA.md)
# FROM glcr.b-data.ch/jupyterlab/cuda/python/scipy
# FROM python:3.11-slim-bookworm
# FROM continuumio/miniconda3
FROM jupyter/minimal-notebook:python-3.10.4

# * Init & Config
ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV TOKENIZERS_PARALLELISM=true

# # Allows "conda create" to restart shell
# SHELL ["bash", "-lc"]

USER root

RUN python3 -m pip install --no-cache-dir --upgrade pip conda
# RUN conda init
RUN conda config --append channels conda-forge

# * Install Basic Packages
# # Prefer conda
RUN conda install \
    gradio \
    transformers \
    datasets \
    gluonts
RUN conda install \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    seqeval \
    tqdm \
    nltk \
    sentencepiece \
    ipykernel \
    ipywidgets \
    mwclient \
    yfinance \
    xgboost \
    gluonts \
    evaluate
RUN conda install -c pytorch pytorch torchvision
RUN conda install -c conda-forge pytorch-lightning pyopencl accelerate
# # Install TinyGrad
RUN python3 -m pip install --no-cache-dir git+https://git@github.com/geohot/tinygrad.git

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
