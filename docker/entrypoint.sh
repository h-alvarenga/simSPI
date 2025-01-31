#!/usr/bin/env bash

set -e
set -o pipefail

export PATH="/work/TEM-simulator/src":$PATH

source /opt/anaconda/etc/profile.d/conda.sh
conda activate base

echo "PATH: $PATH"

ls -ltrh /

echo "Executing command: "

bash -c "set -e; set -o pipefail; $1"