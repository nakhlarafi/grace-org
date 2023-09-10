#!/bin/bash
#SBATCH -J grace_test_jack # job name
#SBATCH --account=r_mdnakh # indicates username (mandatory parameter)
#SBATCH --nodelist=virya3 # specific node to run the job on.
#SBATCH --mem=100M # memory reserved (mandatory parameter)
#SBATCH -o _%x%J.out # output (& error) file name
#SBATCH --mail-type=BEGIN,END # when to send email notification
#SBATCH --mail-user=nakhla054@gmail.com # set email to be used

module load anaconda/3.2022.10
python3 runtotal.py JacksonXml