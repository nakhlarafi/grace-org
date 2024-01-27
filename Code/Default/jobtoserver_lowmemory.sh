#!/bin/bash
#SBATCH -J depgraph-test
#SBATCH --mem=120M
#SBATCH -w virya2
#SBATCH -o depgraph-test.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=nakhla054@gmail.com 

source /etc/profile.d/modules.sh
module load anaconda/3.2023.03
bash job5.sh