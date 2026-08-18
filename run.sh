#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV=script-0caa98bb
if ! conda env list | grep -q "^$ENV "; then
  echo "Creating conda env $ENV..."
  conda env create -f environment.yml -n "$ENV"
fi

conda run -n "$ENV" python run.py
