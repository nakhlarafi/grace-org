#!/bin/bash
#SBATCH -J Test-Closures --mem=100M --gpus=4 -w virya3

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
bash job_cross.sh