<!-- #region -->
# Multi-Granularity Reverse Knowledge Distillation (MG-RKD)

This is a PyTorch implementation of Multi-Granularity Reverse Knowledge Distillation (MG-RKD) which is built on the source code of GLNN (https://github.com/snap-research/graphless-neural-networks/tree/main), and the code includes the following modules:

- Dataset Loader (Cora, Citeseer, Pubmed, Amazon-Photo, Coauthor-CS, Coauthor-Phy)

- Various teacher and student GNN architectures (GCN, GAT, GCN+Initial connection, GAT+Initial connection, MLP)

- Training paradigm for teacher GNNs and student GNNs

- Visualization and evaluation metrics



## Getting Started

### Setup Environment

We use conda for environment setup. You can use `bash ./prepare_env.sh` which will create a conda environment named `rkd` and install relevant requirements (from `requirements.txt`).   For simplicity, we use CPU-based `torch` and `dgl` versions in this guide, as specified in requirements.  To run experiments with CUDA, please install `torch` and `dgl` with proper CUDA support, remove them from `requirements.txt`, and properly set the `--device` argument in the scripts.

Be sure to activate the environment with

`conda activate rkd`

before running experiments as described below.



### Preparing datasets
To run experiments for dataset used in the paper, please download from the following links and put them under `data/` (see below for instructions on organizing the datasets).

*CPF data* (`cora`, `citeseer`, `pubmed`, `a-computer`, and `a-photo`): Download the '.npz' files from [here](https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz). Rename `amazon_electronics_computers.npz` and `amazon_electronics_photo.npz` to `a-computer.npz` and `a-photo.npz` respectively.

## Usage
### To run Classical Node Classification Task
To quickly train a teacher model you can run train_teacher.py by specifying the experiment setting
```
python train_teacher.py --exp_setting tran --teacher GCN --dataset cora
```

To quickly train a student model with a pretrained teacher you can run `train_student.py` by specifying the experiment setting, teacher model, student model, and dataset like the example below. Make sure you train the teacher using the train_teacher.py first and have its result stored in the correct path specified by `--out_t_path`.
```
python train_student.py --exp_setting tran --teacher GCN --student GCN --dataset cora --teacher_num_layers 2 --out_t_path outputs
```
### 





<!-- #endregion -->
