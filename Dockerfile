# Stage 1: Build environment
FROM public.ecr.aws/amazonlinux/amazonlinux:2023-minimal as build

# Set up build arguments and environment variables
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_TARGET_ENV=osml_model_runner

# Set working directory
WORKDIR /home

# Install necessary packages
USER root
RUN dnf update -y && dnf install -y wget shadow-utils gcc && dnf clean all

# Install Miniconda
RUN wget -c ${MINICONDA_URL} && \
    chmod +x ${MINICONDA_VERSION}.sh && \
    ./${MINICONDA_VERSION}.sh -b -f -p /opt/conda && \
    rm ${MINICONDA_VERSION}.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Copy the conda environment file
COPY environment-py310.yml environment.yml

# Create the conda environment and remove additional unnecessary files
RUN conda env create -f environment.yml \
     && conda clean -afy \
     && find /opt/conda/ -follow -type f -name '*.a' -delete \
     && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
     && find /opt/conda/ -follow -type f -name '*.js.map' -delete

# Copy the application source code
COPY . osml-model-runner

# Install the model runner application
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate ${CONDA_TARGET_ENV} && \
    python3 -m pip install osml-model-runner/.

# Stage 2: Runtime environment
FROM public.ecr.aws/amazonlinux/amazonlinux:2023-minimal as model_runner

# Set up runtime environment variables
ENV CONDA_TARGET_ENV=osml_model_runner
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /home

# Copy the conda environment from the build stage
COPY --from=build /opt/conda /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Copy the application from the build stage
COPY --from=build /home/osml-model-runner /home/osml-model-runner

# Create entrypoint script
RUN     (echo '#!/bin/bash' \
    &&   echo '__conda_setup="$(/opt/conda/bin/conda shell.bash hook 2> /dev/null)"' \
    &&   echo 'eval "$__conda_setup"' \
    &&   echo 'conda activate "${CONDA_TARGET_ENV:-base}"' \
    &&   echo '>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"' \
    &&   echo 'exec "$@"'\
        ) >> /entry.sh && chmod +x /entry.sh

# Configure user and permissions
RUN dnf install -y shadow-utils
RUN adduser modelrunner && \
    chown -R modelrunner:modelrunner ./

USER modelrunner

# Set entry point
ENTRYPOINT ["/entry.sh", "/bin/bash", "-c", "python3 osml-model-runner/bin/oversightml-mr-entry-point.py"]
