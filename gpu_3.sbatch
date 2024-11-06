#!/bin/bash
#SBATCH -JGPCR-3
#SBATCH -ot3.out
#SBATCH -Ahive-cs207
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=8G
#SBATCH -t72:00:00
#SBATCH -phive-gpu
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=awallace43@gatech.edu

# VARS TO CHANGE - START
export iter=3
export data_dir=./data_dir/prosmith_test$iter
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

python ./code/preprocessing/preprocessing.py \
    --train_val_path \
    $data_dir/train_val/ \
    --outpath \
    $data_dir/embeddings \
    --smiles_emb_no 2000 --prot_emb_no 2000

echo "preprocessing done"

echo "python ./code/training/training.py \
    --pretrained_model './data_dir/training_data/BindingDB/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl' \
    --train_dir $data_dir/train_val/train.csv \
    --val_dir $data_dir/train_val/test.csv \
    --save_model_path $data_dir/saved_model \
    --embed_path $data_dir/embeddings \
    --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 24 \
    --log_name prosmith_$iter --num_train_epochs 100"

python ./code/training/training.py \
    --pretrained_model "./data_dir/training_data/BindingDB/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl" \
    --train_dir $data_dir/train_val/train.csv \
    --val_dir $data_dir/train_val/test.csv \
    --save_model_path $data_dir/saved_model \
    --embed_path $data_dir/embeddings \
    --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 12 \
    --log_name prosmith_$iter --num_train_epochs 100 \
    --binary_task $binary_task


echo "
python3 -u ./eval.py \
    --train_csv $data_dir/train_val/train.csv \
    --test_csv $data_dir/train_val/test.csv \
    --val_csv $data_dir/train_val/val.csv \
    --pretrained_model $data_dir/saved_model/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
    --embed_path $data_dir/embeddings \
    --binary_task $binary_task
"

python3 -u ./eval.py \
    --train_csv $data_dir/train_val/train.csv \
    --test_csv $data_dir/train_val/test.csv \
    --val_csv $data_dir/train_val/val.csv \
    --pretrained_model $data_dir/saved_model/prosmith_${iter}_1gpus_bs12_1e-05_layers6.txt.pkl \
    --embed_path $data_dir/embeddings \
    --binary_task $binary_task
