import numpy as np
import pandas as pd
import torch
import sys
import json
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig
import os
import torch
import torch.distributed as dist
import logging
import numpy as np

from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
import subprocess
import logging
import pandas as pd
import pickle as pkl
from itertools import accumulate
import random
from time import time

from sklearn import metrics
import matplotlib.pyplot as plt
import argparse


class SMILESProteinDataset(Dataset):
    def __init__(
        self,
        embed_dir,
        data_path,
        train: bool,
        device,
        gpu,
        random_state,
        binary_task: bool,
        extraction_mode=False,
    ):
        start_time = time()
        self.train = train
        self.device = device
        self.gpu = gpu
        self.random_state = random_state
        self.max_prot_seq_len = 1018
        self.max_smiles_seq_len = 256
        self.train_or_test = "train" if train else "test"
        self.binary_task = binary_task
        self.embed_dir = embed_dir

        self.df = pd.read_csv(join(data_path))
        self.prot_dicts = os.listdir(join(embed_dir, "Protein"))
        self.smiles_dicts = os.listdir(join(embed_dir, "SMILES"))
        self.n_prot_dicts = len(self.prot_dicts)
        self.n_smiles_dicts = len(self.smiles_dicts)
        self.num_subsets = self.n_prot_dicts * self.n_smiles_dicts
        self.total_datacount = len(self.df)
        self.data_counts = []
        self.subset_no = 0
        self.protein_subset_no = -1
        self.smiles_subset_no = 0
        self.update_subset()

    def _load_smiles_repr(self, smiles_repr_file):
        with open(smiles_repr_file, "rb") as f:
            smiles_rep = pkl.load(f)
        return smiles_rep

    def _load_protein_repr(self, protein_repr_path):
        map_loc = self.gpu if is_cuda(self.device) else self.device
        return torch.load(protein_repr_path, map_location=map_loc)

    def update_subset(self):
        self.protein_subset_no += 1
        self.protein_subset_no = self.protein_subset_no % self.n_prot_dicts
        self.protein_repr = self._load_protein_repr(
            join(self.embed_dir, "Protein", self.prot_dicts[self.protein_subset_no])
        )

        if self.protein_subset_no == 0:
            smiles_repr_file = join(
                self.embed_dir, "SMILES", self.smiles_dicts[self.smiles_subset_no]
            )
            self.smiles_reprs = self._load_smiles_repr(smiles_repr_file)
            if self.smiles_subset_no < len(self.smiles_dicts) - 1:
                self.smiles_subset_no += 1

        self.subset_no += 1

        all_subset_smiles = list(self.smiles_reprs.keys())
        all_subset_sequences = list(self.protein_repr.keys())

        help_df = self.df.loc[self.df["SMILES"].isin(all_subset_smiles)].copy()
        help_df["index"] = list(help_df.index)
        help_df["Protein sequence"] = [
            seq[:1018] for seq in help_df["Protein sequence"]
        ]
        help_df = help_df.loc[help_df["Protein sequence"].isin(all_subset_sequences)]

        if self.train:
            help_df = help_df.sample(frac=1, random_state=self.random_state)
        help_df = help_df.reset_index(drop=True)

        # logging.info(f"SMILES subset: {self.smiles_subset_no-1}, Protein Subset: {self.protein_subset_no}, Length help_df: {len(help_df)}")
        self.mappings = help_df.copy()

        # logging.info(self.data_counts)
        if len(self.data_counts) == 0:
            self.data_counts.append(len(help_df))
        else:
            self.data_counts.append(self.data_counts[-1] + len(help_df))

    def __len__(self):
        return self.total_datacount

    def __getitem__(self, idx):
        start_time = time()
        """This function assumes lienar data reading"""
        prev_subset_max_idx = (
            0
            if self.subset_no == 1 and self.protein_subset_no == 0
            else self.data_counts[-2]
        )
        curr_subset_max_idx = self.data_counts[-1]
        idx = idx - prev_subset_max_idx
        # logging.info(f"Item: {idx}, len(help_df): {len(self.mappings)}, prev_subset_max_idx : {prev_subset_max_idx}, curr_subset_max_idx : {curr_subset_max_idx}")

        if idx >= len(self.mappings):
            # logging.info(f"updating subset {self.subset_no}")
            self.update_subset()
            while len(self.mappings) == 0:
                self.update_subset()
            prev_subset_max_idx = curr_subset_max_idx
            idx = 0

        label, protein, smiles, index = (
            float(self.mappings["output"][idx]),
            self.mappings["Protein sequence"][idx],
            self.mappings["SMILES"][idx],
            int(self.mappings["index"][idx]),
        )

        if self.binary_task:
            label = int(label)

        smiles_emb = self.smiles_reprs[smiles].squeeze()
        protein_emb = torch.from_numpy(self.protein_repr[protein[:1018]])

        smiles_attn_mask = torch.zeros(self.max_smiles_seq_len)
        smiles_attn_mask[: smiles_emb.shape[0]] = 1
        protein_attn_mask = torch.zeros(self.max_prot_seq_len)
        protein_attn_mask[: protein_emb.shape[0]] = 1

        smiles_padding = (0, 0, 0, self.max_smiles_seq_len - smiles_emb.shape[0])
        prot_padding = (0, 0, 0, self.max_prot_seq_len - protein_emb.shape[0])

        smiles_emb = torch.nn.functional.pad(
            smiles_emb, smiles_padding, mode="constant", value=0
        )
        protein_emb = torch.nn.functional.pad(
            protein_emb, prot_padding, mode="constant", value=0
        )

        labels = torch.Tensor([label])

        labels.requires_grad = False
        smiles_emb = smiles_emb.detach()
        protein_emb = protein_emb.detach()

        return (
            smiles_emb,
            smiles_attn_mask,
            protein_emb,
            protein_attn_mask,
            labels,
            index,
        )


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def is_cuda(device):
    return device == torch.device("cuda")


def accuracy(y_pred, y_true):
    # Calculate accuracy
    correct = (y_pred == y_true).sum().item()
    total = y_true.shape[0]
    acc = correct / total
    return acc


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    ys_orig = np.array(ys_orig).reshape(-1)
    ys_line = np.array(ys_line).reshape(-1)
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


class MM_TNConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        s_hidden_size=767,
        p_hidden_size=1280,
        max_seq_len=1276,
    ):
        self.s_hidden_size = s_hidden_size
        self.p_hidden_size = p_hidden_size
        self.max_seq_len = max_seq_len
        self.binary_task = binary_task

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config


class BertSmilesPooler(nn.Module):
    def __init__(self, config):
        super(BertSmilesPooler, self).__init__()
        self.dense = nn.Linear(config.s_hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertProteinPooler(nn.Module):
    def __init__(self, config):
        super(BertProteinPooler, self).__init__()
        self.dense = nn.Linear(config.p_hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Bert(nn.Module):
    def __init__(self, num_classes, emb_dim, no_layers, binary_task):
        super(Bert, self).__init__()
        self.config = BertConfig(
            hidden_size=emb_dim,
            num_hidden_layers=no_layers,
            num_attention_heads=6,
        )
        self.binary_task = binary_task
        self.num_classes = num_classes
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_hidden_layers = self.config.num_hidden_layers

        # transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    dim_feedforward=4 * self.config.hidden_size,
                    activation="gelu",
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )

        # output layer
        self.hidden_layer = nn.Linear(self.config.hidden_size, 32)
        self.output_layer = nn.Linear(32, num_classes)
        self.sigmoid_layer = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, input_ids, attention_mask, get_repr=False):
        x = input_ids.permute(1, 0, 2)
        # transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, src_key_padding_mask=attention_mask)

        if get_repr:
            hidden_repr = x[0, :, :]

        x = x[0, :, :]
        x = x.reshape(-1, self.hidden_size)
        x = self.hidden_layer(x)

        x = self.ReLU(x)
        x = self.output_layer(x)

        if self.binary_task:
            x = self.sigmoid_layer(x)

        if get_repr:
            return x, hidden_repr
        return x


class MM_TN(nn.Module):
    def __init__(self, config):
        super(MM_TN, self).__init__()

        self.config = config
        self.s_pooler = BertSmilesPooler(config)
        self.p_pooler = BertProteinPooler(config)
        self.main_bert = Bert(
            num_classes=1,
            emb_dim=config.hidden_size,
            no_layers=config.num_hidden_layers,
            binary_task=config.binary_task,
        )

    def extract_repr(self):
        pass

    def forward(
        self,
        smiles_emb,
        smiles_attn,
        protein_emb,
        protein_attn,
        device,
        gpu,
        get_repr=False,
    ):
        batch_size, _, _ = smiles_emb.shape
        s_embedding = self.s_pooler(smiles_emb)
        p_embedding = self.p_pooler(protein_emb)

        zeros_pad = torch.zeros((batch_size, 1, self.config.hidden_size)).to(device)
        zeros_mask = torch.zeros(batch_size, 1).to(device)
        ones_pad = torch.ones((batch_size, 1, self.config.hidden_size)).to(device)
        ones_mask = torch.ones(batch_size, 1).to(device)

        if is_cuda(device):
            zeros_pad, zeros_mask, ones_pad, ones_mask = (
                zeros_pad.cuda(gpu),
                zeros_mask.cuda(gpu),
                ones_pad.cuda(gpu),
                ones_mask.cuda(gpu),
            )

        # <cls> SMILES <sep> Protein
        concat_seq = torch.cat((ones_pad, s_embedding, zeros_pad, p_embedding), dim=1)
        attention_mask = torch.cat(
            (ones_mask, smiles_attn, zeros_mask, protein_attn), dim=1
        )

        if get_repr:
            output, final_repr = self.main_bert(concat_seq, attention_mask, get_repr)
            return output, final_repr

        else:
            output = self.main_bert(concat_seq, attention_mask)
            return output


# --learning_rate 1e-5  --num_hidden_layers 6 --batch_size 24 --binary_task True \
# --log_name HT4_1 --num_train_epochs 100
def load_model(pretrained_model, device, binary_task):
    config = MM_TNConfig.from_dict(
        {
            "s_hidden_size": 600,
            "p_hidden_size": 1280,
            "hidden_size": 768,
            "max_seq_len": 1276,
            "num_hidden_layers": 6,
            "binary_task": binary_task,
        }
    )
    print(f"Loading model")
    model = MM_TN(config)

    if is_cuda(device):
        model = model.to(device)

    if os.path.exists(pretrained_model):
        print(f"Loading model")
        # try:
        state_dict = torch.load(pretrained_model)
        # print(state_dict)
        new_model_state_dict = model.state_dict()
        # print(new_model_state_dict)
        for key in new_model_state_dict.keys():
            mod_key = f"module.{key}"
            new_model_state_dict[key].copy_(state_dict[mod_key])
            # print("Update key: %s" % key)
            # try:
            #     print("Update key: %s" % key)
            # except:
            #     None
        model.load_state_dict(new_model_state_dict)
        print("Successfully loaded pretrained model")
        # except:
        #     new_state_dict = {}
        #     for key, value in state_dict.items():
        #         new_state_dict[key.replace("module.", "")] = value
        #     model.load_state_dict(new_state_dict)
        #     print("Successfully loaded pretrained model (V2)")
        #
    else:
        raise ValueError("Model path is invalid, cannot load pretrained MM_TN model")
    return model


def load_data(
    val_dir, embed_path, binary_task, device, gpu, batch_size=24, world_size=1
):
    val_dataset = SMILESProteinDataset(
        data_path=val_dir,
        embed_dir=embed_path,
        train=False,
        device=device,
        gpu=gpu,
        random_state=42,
        binary_task=binary_task,
        extraction_mode=False,
    )
    valsampler = DistributedSampler(
        val_dataset, shuffle=False, num_replicas=world_size, rank=gpu, drop_last=True
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        sampler=valsampler,
    )
    return valloader


def eval_dataloader(model, dataloader, gpu, device, binary_task):
    criterion = MSELoss()
    y_true, y_pred = [], []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # move batch to device
            if is_cuda(device):
                batch = [r.cuda(gpu) for r in batch]
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels, _ = batch
            # forward pass
            outputs = model(
                smiles_emb=smiles_emb,
                smiles_attn=smiles_attn,
                protein_emb=protein_emb,
                protein_attn=protein_attn,
                device=device,
                gpu=gpu,
            )
            preds = outputs

            if binary_task:
                y_true.extend(labels.cpu().bool().detach().numpy())
                y_pred.extend(preds.cpu().detach().numpy())
            else:
                y_true.extend(labels.cpu().detach().numpy())
                y_pred.extend(preds.cpu().detach().numpy())
    return y_true, y_pred


def plot_results():
    results = np.load("./plots/results.npy")
    y_true = results[:, 0]
    y_pred = results[:, 1]
    true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    false_negative = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negative}")
    # y_true += np.random.normal(0, 0.1, len(y_true))
    # y_pred += np.random.normal(0, 0.02, len(y_pred))
    # print(results)
    vals = len(y_true)
    correct = np.sum(np.equal(y_true, y_pred))
    binding_ligands_cnt_true = np.sum(y_true)
    binding_ligands_cnt_pred = np.sum(y_pred)
    percent_correct = correct / vals
    print(f"correct / total: {correct} / {vals}\nAccuracy: {percent_correct}")
    print(
        f"TRUE binding : non-binding = {binding_ligands_cnt_true} : {vals - binding_ligands_cnt_true}"
    )
    print(
        f"PRED binding : non-binding = {binding_ligands_cnt_pred} : {vals - binding_ligands_cnt_pred}"
    )
    # Plot ROC curve of the model
    display = metrics.RocCurveDisplay.from_predictions(
        y_true,
        y_pred,
    )
    display.plot()
    plt.savefig("./plots/ROC_curve.png", dpi=400)
    return


def evaluate_split_performance(
    model,
    dloader,
    gpu,
    device,
    binary_task,
    plot_ROC_path="./plots/ROC_curve.png",
):
    # print(f"dloader size: {len(dloader)}")
    if binary_task:
        y_true, y_pred = eval_dataloader(model, dloader, gpu, device, binary_task)
        if binary_task:
            y_pred = np.rint(y_pred)
        results = np.hstack((y_true, y_pred))
        y_true = results[:, 0]
        y_pred = results[:, 1]
        # print(results[:10, :])

        true_positives = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        false_positives = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        false_negative = np.sum(np.logical_and(y_true == 1, y_pred == 0))
        print(f"True positives: {true_positives}")
        print(f"False positives: {false_positives}")
        print(f"False negatives: {false_negative}")
        vals = len(y_true)
        correct = np.sum(np.equal(y_true, y_pred))
        binding_ligands_cnt_true = np.sum(y_true)
        binding_ligands_cnt_pred = np.sum(y_pred)
        percent_correct = correct / vals
        print(f"correct / total: {correct} / {vals}\nAccuracy: {percent_correct}")
        print(
            f"TRUE binding : non-binding = {binding_ligands_cnt_true} : {vals - binding_ligands_cnt_true}"
        )
        print(
            f"PRED binding : non-binding = {binding_ligands_cnt_pred} : {vals - binding_ligands_cnt_pred}"
        )
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="example estimator"
        )
        display.plot()
        plt.savefig(plot_ROC_path)
    else:
        y_true, y_pred = eval_dataloader(model, dloader, gpu, device, binary_task)
        results = np.hstack((y_true, y_pred))
        # MSE loss
        loss = np.mean((results[:, 0] - results[:, 1]) ** 2)
        print(f"MSE Loss: {loss}")
        # print(f"R2 score: {r_squared_error(y_true, y_pred)}")
        print(results[:10, :])
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="The input train dataset",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="The input train dataset",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="The input val dataset",
    )
    parser.add_argument(
        "--pretrained_model",
        default='',
        type=str,
        help="Path of pretrained model.",
    )
    parser.add_argument(
        "--binary_task",
        default=False,
        type=bool,
        help="Specifies wether the target variable is binary or continous.",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
        help="Path that contains subfolders SMILES and Protein with embedding dictionaries",
    )
    args = parser.parse_args()
    print(args)
    val_csv = args.val_csv
    train_csv = args.train_csv
    test_csv = args.test_csv
    pretrained_model = args.pretrained_model
    binary_task = args.binary_task
    embed_path = args.embed_path
    gpu = 0
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # device_ids = list(range(torch.cuda.device_count()))
        # gpus = len(device_ids)
        # world_size = gpus
    else:
        device = torch.device("cpu")
        # world_size = -1
    model = load_model(pretrained_model, device, binary_task)
    print("\nVal\n")
    loader_v = load_data(val_csv, embed_path, binary_task, device, gpu)
    evaluate_split_performance(model, loader_v, gpu, device, binary_task)
    print("\nTest\n")
    loader_test = load_data(test_csv, embed_path, binary_task, device, gpu)
    evaluate_split_performance(model, loader_test, gpu, device, binary_task)
    print("\nTrain\n")
    loader_train = load_data(train_csv, embed_path, binary_task, device, gpu)
    evaluate_split_performance(model, loader_train, gpu, device, binary_task)
    return


if __name__ == "__main__":
    main()
