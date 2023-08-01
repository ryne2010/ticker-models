# ! Adaped from Jupyter Dockerstack recommended for Transformers (https://github.com/ToluClassics/transformers_notebook/blob/main/Dockerfile)
FROM jupyter/base-notebook:latest

ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV TOKENIZERS_PARALLELISM=true

# Add RUN statements to install packages as the $NB_USER defined in the base images.

# Add a "USER root" statement followed by RUN statements to install system packages using apt-get,
# change file permissions, etc.

# If you do switch to root, always be sure to add a "USER $NB_USER" command at the end of the
# file to ensure the image runs as a unprivileged user by default.

USER root

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    jupyter \
    # tensorflow-cpu \
    torch \
    torchvision \
    torchaudio
    # jax \
    # jaxlib \
    # optax

RUN python3 -m pip install --no-cache-dir \
    transformers \
    datasets\
    nltk \
    pytorch_lightning \
    gradio \
    sentencepiece \
    seqeval

# * Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN python3 -m pip install --no-cache-dir --requirement /tmp/requirements.txt


USER ${NB_UID}

WORKDIR "${HOME}"