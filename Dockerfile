# Set the base image to build from Internal Amazon Docker Image rather than DockerHub
# If a lot of request were made, CodeBuild will failed due to...
# "You have reached your pull rate limit. You may increase the limit by authenticating and upgrading"
ARG BASE_CONTAINER=public.ecr.aws/amazonlinux/amazonlinux:latest

# Swap BASE_CONTAINER to a container output while building cert-base if you need to override the pip mirror
FROM ${BASE_CONTAINER} as model_runner

# Only override if you're using a mirror with a cert pulled in using cert-base as a build parameter
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/

# Define required packages to install
ARG PACKAGES="wget"

# Give sudo permissions
USER root

# Configure, update, and refresh yum enviornment
RUN yum update -y && yum clean all && yum makecache

# Install all our dependancies
RUN yum install -y $PACKAGES

# Install miniconda
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /usr/local

# Clean up installer file
RUN rm ${MINICONDA_VERSION}.sh

# Install GDAL and python venv to the user profile
# This sets the python3 alias to be the miniconda managed python3.10 ENV
ARG PYTHON_VERSION=3.10
RUN conda install -c conda-forge  -q -y --prefix /usr/local python=${PYTHON_VERSION} gdal

ARG INSTALL_DIR=/home/osml-model-runner

# Copy all our model runner source code
COPY . ${INSTALL_DIR}/

RUN chmod +x --recursive /home
RUN chmod 777 --recursive /home

# Import the source directory to the generalized path
# ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_DIR}/src"

# Hop in the home directory
WORKDIR ${INSTALL_DIR}

# Install package module to the instance
RUN python3 -m pip install .

# Clean up any dangling conda resources
RUN conda clean -afy

# Set the user to the ecs-provisioned user
USER 1000

# Set the entry point command to start model runner
CMD ["python3", "bin/oversightml-mr-entry-point.py"]

FROM model_runner as unit-test

# Hop in the home directory
WORKDIR ${INSTALL_DIR}

# Set root user for dep installs
USER root

# Import the source directory to the generalized path
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_DIR}/src/"

# Set the entry point command to run unit tests
CMD ["python3", "-m", "pytest","--cov=aws.osml.model_runner", "test/"]
