#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate aes
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python3 main.py --simclr --mnist --val
