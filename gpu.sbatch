#!/bin/bash
#SBATCH -JGPCR
#SBATCH -ot1.out
#SBATCH -Ahive-cs207
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH -phive-gpu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


cd $HOME/data/gits/ProSmith
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate /storage/hive/project/chem-sherrill/awallace43/.conda/envs/orml

echo "starting job..."
bash ./preprocess.sh
echo "preprocessing done"
bash ./orml_train.sh
echo "finished job!"
