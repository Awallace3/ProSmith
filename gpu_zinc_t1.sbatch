#!/bin/bash
#SBATCH -Jzinc_screen1
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -ozinc-1.out
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH -pgpu-v100
#SBATCH -Agts-cs207-chemx
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


# VARS TO CHANGE - START
export iter=1
export data_dir=/storage/coda1/p-cs207/0/awallace43/ProSmith/data_dir
export binary_task=False
# VARS TO CHANGE - END


cd /storage/coda1/p-cs207/0/awallace43/ProSmith
conda init
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/prosmith


echo "
python ./code/preprocessing/preprocessing.py \
    --train_val_path ${data_dir}/csvs \
    --outpath ${data_dir}/zinc_embeddings_${iter} \
    --smiles_emb_no 2000 --prot_emb_no 2000
"

python ./code/preprocessing/preprocessing.py \
    --train_val_path ${data_dir}/csvs \
    --outpath ${data_dir}/zinc_embeddings_${iter} \
    --smiles_emb_no 2000 --prot_emb_no 2000

echo "preprocessing done"


echo "
python3 -u ./eval_zinc.py \
    --val_csv ${data_dir}/csvs/zinc_ht4.csv \
    --pretrained_model $data_dir/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
    --embed_path ${data_dir}/zinc_embeddings_${iter} \
    --binary_task $binary_task
"

python3 -u ./eval_zinc.py \
    --val_csv ${data_dir}/csvs/zinc_ht4.csv \
    --pretrained_model $data_dir/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
    --embed_path ${data_dir}/zinc_embeddings_${iter} \
    --binary_task $binary_task \
    --roc_plot_path prosmith_${iter}_zinc_roc.png \
    --output_csv zinc_5ht4_t${iter}.csv 
