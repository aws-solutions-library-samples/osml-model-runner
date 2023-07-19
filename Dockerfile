FROM public.ecr.aws/amazonlinux/amazonlinux:2023 as model_runner

# only override if you're using a mirror with a cert pulled in using cert-base as a build parameter
ARG BUILD_CERT=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
ARG PIP_INSTALL_LOCATION=https://pypi.org/simple/

# give sudo permissions
USER root

# set working directory to home
WORKDIR /home

# configure, update, and refresh yum enviornment
RUN yum update -y && yum clean all && yum makecache && yum install -y wget shadow-utils

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

# set all the ENV vars needed for build
ENV CONDA_TARGET_ENV=osml_model_runner
ENV CC="clang"
ENV CXX="clang++"
ENV ARCHFLAGS="-arch x86_64"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib/:/opt/conda/bin:/usr/include:/usr/local/"
ENV PROJ_LIB=/opt/conda/share/proj

# copy our conda env configuration for Python 3.10
COPY environment-py310.yml environment.yml

# create the conda env
RUN conda env create

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

# configure .bashrc to drop into a conda env and immediately activate our TARGET env
RUN conda init && echo 'conda activate "${CONDA_TARGET_ENV:-base}"' >>  ~/.bashrc

# copy our lcoal application source into the container
COPY . osml-model-runner

# install the model runner application from source
RUN python3 -m pip install osml-model-runner/

# clean up the conda install
RUN conda clean -afy

# set up a health check at that port
HEALTHCHECK NONE

# set up a user to run the container as and assume it
RUN adduser modelrunner
RUN chown -R modelrunner:modelrunner ./
USER modelrunner

# set the entry point script
ENTRYPOINT ["/entry.sh", "/bin/bash", "-c", "python3 osml-model-runner/bin/oversightml-mr-entry-point.py"]
