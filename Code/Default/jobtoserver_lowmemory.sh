#!/bin/bash
#SBATCH -J Depgrah --mem=100M --gpus=1 -w virya4

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
bash job5.sh