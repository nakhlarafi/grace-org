#!/bin/bash
#SBATCH -J Test-GPUs --mem=100M --gpus=4 -w virya4
nvidia-smi -L