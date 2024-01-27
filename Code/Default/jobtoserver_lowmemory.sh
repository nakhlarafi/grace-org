#!/bin/bash
#SBATCH -J grace_test_jack 
#SBATCH -w virya3
#SBATCH --mem=100M 
#SBATCH -o _%x%J.txt

source /etc/profile.d/modules.sh 
module load anaconda/3.2022.10
python runtotal.py Collections
# python runtotal.py JacksonXml