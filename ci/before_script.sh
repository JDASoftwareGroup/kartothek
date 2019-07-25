#!/bin/bash

set -xeo pipefail
  
pip freeze

pip install pre-commit && pre-commit run -a

python setup.py docs