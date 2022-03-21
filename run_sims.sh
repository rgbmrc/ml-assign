#! /usr/bin/env bash
source $HOME/tools/miniconda/etc/profile.d/conda.sh
conda activate ml
printf 'n%s\n' {0..19} | shuf | xargs -P8 -i python scripts/train.py 1A {} 2
printf 'n%s\n' {0..15} | shuf | xargs -P8 -i python scripts/train.py 1B {} 2
printf 'n%s\n' {0..19} | shuf | xargs -P8 -i python scripts/train.py 2A {} 2
printf 'n%s\n' {0..19} | shuf | xargs -P8 -i python scripts/train.py 2B {} 2
printf 'n%s\n' {0..19} | shuf | xargs -P8 -i python scripts/train.py 2C {} 2
printf 'n%s\n' {0..19} | shuf | xargs -P8 -i python scripts/train.py 3A {} 2
