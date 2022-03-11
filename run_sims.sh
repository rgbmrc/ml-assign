#! /usr/bin/env bash
set -e
source $HOME/tools/miniconda/etc/profile.d/conda.sh
conda activate ml
printf 'n%s\n' {0..119} | shuf | xargs -P8 -i python scripts/train.py 1 {} 1
