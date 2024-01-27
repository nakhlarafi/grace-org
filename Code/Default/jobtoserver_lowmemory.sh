#!/bin/bash
#SBATCH -J Depgrah --mem=100M --gpus=1 -w virya3

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
bash job5.sh