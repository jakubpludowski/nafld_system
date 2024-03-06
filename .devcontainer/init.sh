#!/bin/bash

# Git config Windows compatibility
REPO=$PWD
(cd / && git config --file $REPO/.git/config core.worktree ..)
git config core.fileMode false
git config core.autocrlf input
git config credential.useHttpPath true
git config core.editor "code --wait"

git config --global user.name "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"

# Poetry available by default in VS Code terminal
sudo ln -s $HOME/.local/bin/poetry /bin/poetry

# Dependencies and tools
poetry self update
poetry install --no-interaction
poetry run pre-commit install --install-hooks

PYTHON_PATH=`find /home/vscode/.cache/pypoetry/virtualenvs/ -path "*/bin/python"`
mkdir .vscode
echo '{
    "python.pythonPath": '\"${PYTHON_PATH}\"'
}' >  .vscode/settings.json
