#! /usr/bin/env bash
set -e
source $HOME/tools/miniconda/etc/profile.d/conda.sh
conda activate ml
printf 'n%s\n' {0..24} | shuf | xargs -P4 -i python scripts/train.py 2A {} 2
