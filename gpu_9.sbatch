#!/bin/bash
#SBATCH -JGPCR-9
#SBATCH -ot9.out
#SBATCH -Ahive-cs207
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH -phive-gpu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu
#SBATCH --exclude=atl1-1-01-018-27-0


# VARS TO CHANGE - START
export iter=9
export data_dir=./data_dir/prosmith_test$iter
export binary_task=False
# VARS TO CHANGE - END

cd $HOME/data/gits/gpcr_ml/phillip/
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate /storage/hive/project/chem-sherrill/awallace43/.conda/envs/prosmith

cd ../ProSmith/

for i in {0..4}; do
    echo "Processing split $i"
    echo "
    python ./code/preprocessing/preprocessing.py \
        --train_val_path \
        $data_dir/train_val_split_${i}/ \
        --outpath \
        $data_dir/embeddings_split_${i} \
        --smiles_emb_no 2000 --prot_emb_no 2000
    "

    python ./code/preprocessing/preprocessing.py \
        --train_val_path \
        $data_dir/train_val_split_${i}/ \
        --outpath \
        $data_dir/embeddings_split_${i} \
        --smiles_emb_no 2000 --prot_emb_no 2000

    echo "preprocessing done"
    #
    echo "python ./code/training/training.py \
        --pretrained_model ./data_dir/t4_models/t4_epoch_4/prosmith_4_4gpus_bs48_1e-05_layers6.txt.pkl \
        --train_dir $data_dir/train_val_split_${i}/train.csv \
        --val_dir $data_dir/train_val_split_${i}/test.csv \
        --save_model_path $data_dir/saved_model_split_${i} \
        --embed_path $data_dir/embeddings_split_${i} \
        --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 24 \
        --log_name prosmith_$iter --num_train_epochs 100"

    python ./code/training/training.py \
        --pretrained_model ./data_dir/t4_models/t4_epoch_4/prosmith_4_4gpus_bs48_1e-05_layers6.txt.pkl \
        --train_dir $data_dir/train_val_split_${i}/train.csv \
        --val_dir $data_dir/train_val_split_${i}/test.csv \
        --save_model_path $data_dir/saved_model_split_${i} \
        --embed_path $data_dir/embeddings_split_${i} \
        --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 12 \
        --log_name prosmith_$iter --num_train_epochs 100 \
        --binary_task $binary_task


    echo "
    python3 -u ./eval.py \
        --train_csv $data_dir/train_val_split_${i}/train.csv \
        --test_csv  $data_dir/train_val_split_${i}/test.csv \
        --val_csv   $data_dir/train_val_split_${i}/val.csv \
        --pretrained_model $data_dir/saved_model_split_${i}/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
        --embed_path $data_dir/embeddings_split_${i} \
        --binary_task $binary_task
    "

    python3 -u ./eval.py \
        --train_csv $data_dir/train_val_split_${i}/train.csv \
        --test_csv  $data_dir/train_val_split_${i}/test.csv \
        --val_csv   $data_dir/train_val_split_${i}/val.csv \
        --pretrained_model $data_dir/saved_model_split_${i}/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
        --embed_path $data_dir/embeddings_split_${i} \
        --binary_task $binary_task \
        --roc_plot_path ../phillip/plots/roc/prosmith_${iter}/split_${i}_roc.png 
done
