#!/bin/bash
#SBATCH -J Test-GPUs --mem=100M --gpus=4 -w virya4

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
python runtotal.py Closure