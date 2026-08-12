import numpy as np
import torch
import logging
import pytz
import random
import os
import yaml
import shutil
from datetime import datetime
from ogb.nodeproppred import Evaluator
from dgl import function as fn
from flax.core.frozen_dict import FrozenDict 

from contextlib import contextmanager
from flax import traverse_util
from flax.core import freeze, unfreeze
from jax import tree_map # random,
from jax.tree_util import tree_reduce

rngmix = lambda rng, x: random.fold_in(rng, hash(x))

CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_config(config_path, model_name, dataset):#从 train.conf.yaml 文件中加载模型配置
    with open(config_path, "r") as conf:#只读模式打开指定文件，返回一个文件对象 conf
        full_config = yaml.load(conf, Loader=yaml.FullLoader)#使用 PyYAML 库加载文件内容，将 YAML 格式的配置文件解析成 Python 字典。yaml.FullLoader 是一种用于解析 YAML 文件的解析器，能够加载所有类型的 YAML 数据。
   #通过 yaml.load() 函数，full_config是一个包含整个配置文件内容的Python字典。
    dataset_specific_config = full_config["global"]
    #从 full_config 字典中获取键 "global" 对应的值，保存到变量 dataset_specific_config 中。
    model_specific_config = full_config[dataset][model_name] 

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)#合并
    else:
        specific_config = dataset_specific_config

    specific_config["model_name"] = model_name
    return specific_config


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def check_readable(path):
    if not os.path.exists(path):
        raise ValueError(f"No such file or directory! {path}")


def timetz(*args):
    tz = pytz.timezone("Asia/Shanghai")
    return datetime.now(tz).timetuple()


def get_logger(filename, console_log=False, log_level=logging.INFO):#是否在控制台输出日志（默认 False）
    tz = pytz.timezone("Asia/Shanghai")
    log_time = datetime.now(tz).strftime("%b%d_%H_%M_%S")
    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging防止日志信息传播到父记录器
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers移除所有现有的处理器，避免重复添加
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:#如果 console_log 为 True，则在控制台输出日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def idx_split(idx, ratio, seed=0):#将索引数组按比例随机分割为两部分
    """
    randomly split idx into two portions with ratio% elements and (1 - ratio)% elements
    ratio:第一部分的占比
    """
    set_seed(seed)#确保结果可复现
    n = len(idx) #索引数组长度
    cut = int(n * ratio) #第一部分长度
    idx_idx_shuffle = torch.randperm(n) #生成一个随机排列的索引数组

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    #按比例分割成两部分
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """
    将图数据的测试集进一步划分为传导测试集(idx_test_tran)和归纳测试集(idx_test_ind)，并返回观测子图的索引
    rate:归纳测试集占测试集的比例
    Args:
        The original setting was transductive传导. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)归纳评估要隐藏多少测试

        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18
        rate = 18 / 90

    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,可观察子图
        where as indices start directly with 'idx_' correspond to the node indices in the original graph,原始图
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])#将原始训练集、验证集和tran测试集合并为观测子图的索引
    N1, N2 = idx_train.shape[0], idx_val.shape[0]#子图中训练、验证集
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1] #0-N1
    obs_idx_val = obs_idx_all[N1 : N1 + N2] #N1-N1+N2
    obs_idx_test = obs_idx_all[N1 + N2 :] #N1+N2-最后

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def get_evaluator(dataset):
    if dataset in CPF_data + NonHom_data + BGNN_data:

        def evaluator(out, labels):
            pred = out.argmax(1) #获取预测类别
            return pred.eq(labels).float().mean().item() #计算准确率

    elif dataset in OGB_data:
        ogb_evaluator = Evaluator(dataset)

        def evaluator(out, labels):
            pred = out.argmax(1, keepdim=True)
            input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
            return ogb_evaluator.eval(input_dict)["acc"]

    else:
        raise ValueError("Unknown dataset")

    return evaluator #返回一个评估器


def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

    return evaluator


def compute_min_cut_loss(g, out):
    out = out.to("cpu")
    S = out.exp()
    A = g.adj().to_dense()
    D = g.in_degrees().float().diag()
    min_cut = (
        torch.matmul(torch.matmul(S.transpose(1, 0), A), S).trace()
        / torch.matmul(torch.matmul(S.transpose(1, 0), D), S).trace()
    )
    return min_cut.item()


def feature_prop(feats, g, k):
    """
    Augment node feature by propagating the node features within k-hop neighborhood.
    The propagation is done in the SGC fashion, i.e. hop by hop and symmetrically normalized by node degrees.
    """
    assert feats.shape[0] == g.num_nodes()

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5).unsqueeze(1)

    # compute (D^-1/2 A D^-1/2)^k X
    for _ in range(k):
        feats = feats * norm
        g.ndata["h"] = feats
        g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        feats = g.ndata.pop("h")
        feats = feats * norm

    return feats

def flatten_params(params):
  return {"/".join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}

def unflatten_params(flat_params):
  return freeze(
      traverse_util.unflatten_dict({tuple(k.split("/")): v
                                    for k, v in flat_params.items()}))

def merge_params(a, b):
  return unflatten_params({**a, **b})

def lerp(lam, t1, t2):
    t1 = unfreeze(t1) if isinstance(t1, FrozenDict) else t1
    t2 = unfreeze(t2) if isinstance(t2, FrozenDict) else t2
    return tree_map(lambda a, b: (1 - lam) * a + lam * b, t1, t2)