#!/bin/bash
#SBATCH -J Test-GPUs --mem=100M --gpus=4 -w virya4
source /etc/profile.d/modules.sh 
module load anaconda/3.2023.10
python runtotal.py Cli
python runtotal.py Codec
python runtotal.py Compress
python runtotal.py Csv
python runtotal.py Gson
python runtotal.py JacksonCore
python runtotal.py JacksonXml
python runtotal.py Jsoup
python runtotal.py Math
python runtotal.py Mockito
python runtotal.py Time
