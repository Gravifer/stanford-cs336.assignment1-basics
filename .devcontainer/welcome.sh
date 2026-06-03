#!/usr/bin/env bash
 echo "Welcome to the Dev Container! Customize it at .devcontainer/devcontainer.json" \
 | sudo tee /usr/local/etc/vscode-dev-containers/first-run-notice.txt > /dev/null
if [ -d /workspaces/.codespaces/shared ]; then
 echo "Welcome to the Codespace! Customize it at .devcontainer/devcontainer.json" \
 | sudo tee /workspaces/.codespaces/shared/first-run-notice.txt       > /dev/null
fi