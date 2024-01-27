#!/bin/bash
#SBATCH --mem=100M -n1 --gpus=1 -J depgraphjob

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
bash job5.sh