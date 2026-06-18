#!/usr/bin/env bash
# One-time Pi layout for AFLFantasy-DNN (run on the Pi).
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-delophant}"

sudo mkdir -p /srv/aflfantasy/prod
sudo chown -R "${DEPLOY_USER}:${DEPLOY_USER}" /srv/aflfantasy

echo "Created /srv/aflfantasy/prod (owner: ${DEPLOY_USER})"
echo "Next: git clone into /srv/aflfantasy/prod, then create venv with requirements-rpi.txt"
