#!/bin/bash
#SBATCH -J Test-GPUs --mem=100M --gpus=4 -w virya4
source /etc/profile.d/modules.sh 
module load anaconda/3.2022.10
python runtotal.py Collections