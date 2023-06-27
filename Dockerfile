# set the base image to build from Internal Amazon Docker Image rather than DockerHub
# if a lot of request were made, CodeBuild will failed due to...
# "You have reached your pull rate limit. You may increase the limit by authenticating and upgrading"
ARG BASE_CONTAINER=public.ecr.aws/amazonlinux/amazonlinux:latest

# swap BASE_CONTAINER to a container output while building cert-base if you need to override the pip mirror
FROM ${BASE_CONTAINER} as model_runner

# only override if you're using a mirror with a cert pulled in using cert-base as a build parameter
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/

# give sudo permissions
USER root

# set working directory to home
WORKDIR /home/

# configure, update, and refresh yum enviornment
RUN yum update -y && yum clean all && yum makecache && yum install -y wget

# install miniconda
ARG MINICONDA_VERSION=Miniconda3-latest-Linux-x86_64
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/${MINICONDA_VERSION}.sh
RUN wget -c ${MINICONDA_URL} \
    && chmod +x ${MINICONDA_VERSION}.sh \
    && ./${MINICONDA_VERSION}.sh -b -f -p /opt/conda \
    && rm ${MINICONDA_VERSION}.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# add conda to the path so we can execute it by name
ENV PATH=/opt/conda/bin:$PATH

# target conda env for container
ENV CONDA_TARGET_ENV=osml_model_runner

# create /entry.sh which will be our new shell entry point
# this performs actions to configure the environment
# before starting a new shell (which inherits the env).
# the exec is important as this allows signals to passpw
RUN     (echo '#!/bin/bash' \
    &&   echo '__conda_setup="$(/opt/conda/bin/conda shell.bash hook 2> /dev/null)"' \
    &&   echo 'eval "$__conda_setup"' \
    &&   echo 'conda activate "${CONDA_TARGET_ENV:-base}"' \
    &&   echo '>&2 echo "ENTRYPOINT: CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"' \
    &&   echo 'exec "$@"'\
        ) >> /entry.sh && chmod +x /entry.sh

# tell the docker build process to use this for RUN.
# the default shell on Linux is ["/bin/sh", "-c"], and on Windows is ["cmd", "/S", "/C"]
SHELL ["/entry.sh", "/bin/bash", "-c"]

# copy our conda env configuration
COPY environment.yml .

# create the conda env
RUN conda env create

# configure .bashrc to drop into a conda env and immediately activate our TARGET env
RUN conda init && echo 'conda activate "${CONDA_TARGET_ENV:-base}"' >>  ~/.bashrc

# copy our lcoal application source into the container
COPY . .

# install the application from source
RUN python3 -m pip install .

# clean up the conda install
RUN conda clean -afy

# set the entry point script
ENTRYPOINT ["/entry.sh"]

# set the entry point command to start model runner in the conda env
CMD python3 bin/oversightml-mr-entry-point.py
