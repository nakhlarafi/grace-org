#!/bin/bash
#SBATCH -J grace-closure-pruned --mem=120GB --account=r_mdnakh --gpus=1
#SBATCH --mail-type=BEGIN,END 
#SBATCH --mail-user=nakhla054@gmail.com 

# Array of MIDs
MIDs=("Cli" "Codec" "Collections" "Compress" "Csv" "Gson" "JacksonCore" "JacksonXml" "Jsoup" "Math" "Mockito")

# List of project names
projects=("Cli" "Codec" "Collections" "Compress" "Csv" "Gson" "JacksonCore" "JacksonXml" "Jsoup" "Lang" "Math" "Mockito" "Time")

# Loop through each MID
for MID in "${MIDs[@]}"; do
    echo "Processing MID: ${MID}"

    # Loop through each project for the current MID
    for project in "${projects[@]}"; do
        echo "Running tests for project ${project} with MID ${MID}"
        python run_test.py ${project} ${MID}
        python top_k.py ${project} ${MID}
    done
done
