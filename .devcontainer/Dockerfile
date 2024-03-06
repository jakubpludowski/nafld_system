# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.163.1/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends apt-transport-https ca-certificates wget dirmngr gnupg software-properties-common lsb-release \
    #
    # Setup for Java
    && mkdir -p /usr/share/man/man1 \
    && wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/trusted.gpg.d/adoptium.asc \
    && echo "deb https://packages.adoptium.net/artifactory/deb/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/adoptium.list \
    #
    # Setup for Azure CLI
    && wget -qO - https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /etc/apt/trusted.gpg.d/microsoft.gpg > /dev/null \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/azure-cli.list \
    #
    # Install
    && apt-get update \
    && apt-get -y install --no-install-recommends temurin-8-jdk \
    && apt-get -y install azure-cli \
    && apt-get autoremove -y \
    && apt-get clean -y

USER vscode

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
