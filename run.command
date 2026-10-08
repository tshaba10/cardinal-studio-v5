#!/bin/zsh
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Follow README.md to create the project environment first.'
  exit 1
fi
exec ./.venv/bin/python -m streamlit run app.py "$@"
