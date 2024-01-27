#!/bin/bash
#SBATCH -J Test-GPUs
#SBATCH --mem=10GB 
#SBATCH -o _%x%J.out
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN,END,FAIL --mail-user=nakhla054@gmail.com

source /etc/profile.d/modules.sh 
module load anaconda/3.2023.03
bash job5.sh