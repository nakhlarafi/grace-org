#!/bin/bash
#SBATCH -J Test-GPUs --mem=100M --account=r_mdnakh --gpus=40gb:1
nvidia-smi -L
source /etc/profile.d/modules.sh
module load anaconda/3.2022.10
python runtotal.py JacksonXml