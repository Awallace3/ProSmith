#!/bin/bash
#SBATCH -JGPCR-ALL
#SBATCH -oeval_all.out
#SBATCH -Ahive-cs207
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH -phive-gpu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu


# VARS TO CHANGE - START
export data_dir_va=./data_dir/prosmith_test7
export binary_task=False
# VARS TO CHANGE - END

cd $HOME/data/gits/gpcr_ml/phillip/
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate /storage/hive/project/chem-sherrill/awallace43/.conda/envs/prosmith

echo "python3 -u main.py --iter $iter"

python3 -u main.py --iter $iter
cd ../ProSmith/

echo "
python ./code/preprocessing/preprocessing.py \
    --train_val_path \
    $data_dir/train_val/ \
    --outpath \
    $data_dir/embeddings \
    --smiles_emb_no 2000 --prot_emb_no 2000
"

for iter in 1 2 3 4 5 6 7 8 9 10; do
    export data_dir=./data_dir/prosmith_test${iter}
    echo "
    python3 -u ./eval.py \
        --train_csv $data_dir_val/train_val/train.csv \
        --test_csv $data_dir_val/train_val/test.csv \
        --val_csv $data_dir_val/train_val/val.csv \
        --pretrained_model $data_dir/saved_model/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
        --embed_path $data_dir_val/embeddings \
        --binary_task $binary_task
    "

    python3 -u ./eval.py \
        --train_csv $data_dir_val/train_val/train.csv \
        --test_csv $data_dir_val/train_val/test.csv \
        --val_csv $data_dir_val/train_val/val.csv \
        --pretrained_model $data_dir/saved_model/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
        --embed_path $data_dir_val/embeddings \
        --binary_task $binary_task \
        --roc_plot_path ../phillip/plots/roc/prosmith_${iter}_roc.png 
done
