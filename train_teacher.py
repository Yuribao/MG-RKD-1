import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from models import Model
from dataloader import load_data
from utils import (
    get_logger,
    get_evaluator,
    set_seed,
    get_training_config,
    check_writable,
    compute_min_cut_loss,
    graph_split,
    feature_prop,
)
from train_and_eval import run_transductive, run_inductive


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--device", type=int, default=0, help="CUDA device, -1 means CPU")
    # parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--log_level",
        type=int,
        default=20,
        help="Logger levels for run {10: DEBUG, 20: INFO, 30: WARNING}",
    )
    parser.add_argument(#设置为True以在控制台中显示日志信息
        "--console_log",
        action="store_true",
        help="Set to True to display log info in console",
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Path to save outputs"
    )
    parser.add_argument(
        "--num_exp", type=int, default=1, help="Repeat how many experiments"
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default="tran",
        help="Experiment setting, one of [tran, ind]",
    )
    parser.add_argument(#每多少个epoch计算一次
        "--eval_interval", type=int, default=1, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Set to True to save the loss curves, trained model, and min-cut loss for the transductive setting",
    )

    """Dataset"""
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument( #设置训练集中每类数据的标签数量
        "--labelrate_train",
        type=int,
        default=20,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(#设置验证集中每类数据的标签数量
        "--labelrate_val",
        type=int,
        default=30,
        help="How many labeled data per class in valid set",
    )


    parser.add_argument( 
        "--ratio_train",
        type=float,
        default=0.6,
        help="How many labeled data per class as train set",
    )
    parser.add_argument(
        "--ratio_val",
        type=float,
        default=0.2,
        help="How many labeled data per class in valid set",
    )

    parser.add_argument(
        "--split_idx",
        type=int,
        default=0,
        help="For Non-Homo datasets only, one of [0,1,2,3,4]",
    )

    """Model"""
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./train.conf.yaml",
        help="Path to model configeration",
    )
    parser.add_argument("--num_teacher", type=int, default="1", help="Teacher number")
    
    parser.add_argument("--teacher", type=str, nargs="+", default=["GCN"], help="Teacher model")

    # parser.add_argument(
    #     "--lamb",
    #     type=float,
    #     default=0.5,
    #     help="Parameter balances loss from hard labels and teacher outputs, take values in [0, 1]",
    # )
    
    # parser.add_argument(
    #     "--num_layers", type=int, default=4, help="Model number of layers"
    # )
    # parser.add_argument(
    #     "--hidden_dim", type=int, default=128, help="Model hidden layer dimensions"
    # )
    
    # parser.add_argument("--dropout_ratio", type=float, default=0.8)
    parser.add_argument(
        "--norm_type", type=str, default="none", help="One of [none, batch, layer]"
    )

    """SAGE Specific"""
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--fan_out",
        type=str,
        default="5,5",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for sampler"
    )

    """Optimization"""
    #parser.add_argument("--learning_rate", type=float, default=1e-5)#0.01
    parser.add_argument("--weight_decay", type=float, default=0.0005) #0.0005
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="Evaluate once per how many epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Early stop is the score on validation set does not improve for how many epochs",
    )

    """Ablation"""
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0,
        help="add white noise to features for analysis, value in [0, 1] for noise level",
    )
    parser.add_argument(#rate = idx_test_ind : idx_test
        "--split_rate",
        type=float,
        default=0.2,
        help="Rate for graph split, see comment of graph_split for more details",
    )
    parser.add_argument(
        "--compute_min_cut",
        action="store_true",
        help="Set to True to compute and store the min-cut loss",
    )
    parser.add_argument(
        "--feature_aug_k",
        type=int,
        default=0,
        help="Augment node futures by aggregating feature_aug_k-hop neighbor features",
    )

    args = parser.parse_args()
    return args


def run(args, num_layers, hidden_dim, dropout_ratio,learning_rate, lamb, seed):
    """
    Returns:
    score_lst: a list of evaluation results on test set.
    len(score_lst) = 1 for the transductive setting.
    len(score_lst) = 2 for the inductive/production setting.
    """

    """ Set seed, device, and logger """
    set_seed(seed)
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:
        device = "cpu"

    if args.feature_noise != 0 and seed == 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "noisy_features", f"noise_{args.feature_noise}"
        )

    if args.feature_aug_k > 0 and seed == 0:
        args.output_path = Path.cwd().joinpath(
            args.output_path, "aug_features", f"aug_hop_{args.feature_aug_k}"
        )
        args.teacher = f"GA{args.feature_aug_k}{args.teacher}"

    teacher_names = "_".join(args.teacher) #列表转换为字符串
    if args.exp_setting == "tran":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "transductive",
            args.dataset,
            #args.teacher,
            teacher_names,
            f"{num_layers}"
            #f"seed_{seed}",
        )
    elif args.exp_setting == "ind":
        output_dir = Path.cwd().joinpath(
            args.output_path,
            "inductive",
            f"split_rate_{args.split_rate}",
            args.dataset,
            #args.teacher,
            teacher_names,
            f"seed_{seed}",
        )
    else:
        raise ValueError(f"Unknown experiment setting! {args.exp_setting}")
    args.output_dir = output_dir

    check_writable(output_dir, overwrite=False)
    logger = get_logger(output_dir.joinpath("log"), args.console_log, args.log_level)#日志配置
    logger.info(f"output_dir: {output_dir}")

    """ Load data """
    CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo", "ms_academic_cs", "ms_academic_phy"]
    Hetero_data = ["actor", "texas", "cornell", "wisconsin"]



    if args.dataset in CPF_data:
        g, labels, idx_train, idx_val, idx_test = load_data(#载入数据
            args.dataset,
            args.data_path,
            split_idx=args.split_idx,
            seed=seed,
            labelrate_train=args.labelrate_train,#训练集和验证集标签数量
            labelrate_val=args.labelrate_val,
            ratio_train=args.ratio_train,#训练集和验证集标签数量
            ratio_val=args.ratio_val,
        )
    elif args.dataset in Hetero_data:
        g, labels, idx_train, idx_val, idx_test = load_data(#载入数据
            args.dataset,
            args.data_path,
            split_idx=args.split_idx,
            seed=seed,
            ratio_train=args.ratio_train,#训练集和验证集标签数量
            ratio_val=args.ratio_val,
        )

    logger.info(f"Total {g.number_of_nodes()} nodes.")
    logger.info(f"Total {g.number_of_edges()} edges.")

    feats = g.ndata["feat"].to(device) #从图数据结构中获取节点的特征矩阵
    args.feat_dim = g.ndata["feat"].shape[1] #输入特征维度=特征矩阵的列数
    args.label_dim = labels.int().max().item() + 1 #输出特征维度 标签的最大值加 1 
    

    
    #max().item()：将最大值转换为 Python 标量
    if 0 < args.feature_noise <= 1:
        feats = (
            1 - args.feature_noise
        ) * feats + args.feature_noise * torch.randn_like(feats)

    out_avg = None  
    out_emb = None  
    for i in range(args.num_teacher):
        teacher_model = args.teacher[i]  
        logger.info(f"Training teacher model {i+1}: {teacher_model}")

        # 设置教师模型的输出路径
        teacher_output_dir = output_dir.joinpath(teacher_model)
        teacher_output_dir.mkdir(parents=True, exist_ok=True)

        """ Model config """
        conf = {}
        if args.model_config_path is not None:
            conf = get_training_config(args.model_config_path, teacher_model, args.dataset)
        conf = dict(args.__dict__, **conf)
        conf["device"] = device
        conf["teacher"] = teacher_model  # 设置当前教师模型类型
        if num_layers is not None:
            conf["num_layers"] = num_layers
        if hidden_dim is not None:
            conf["hidden_dim"] = hidden_dim
        if dropout_ratio is not None:
            conf["dropout_ratio"] = dropout_ratio
        if lamb is not None:
            conf["lamb"] = lamb
        if seed is not None:
            conf["seed"] = seed
        if learning_rate is not None:
            conf["learning_rate"] = learning_rate
            
        logger.info(f"conf: {conf}")

        model = Model(conf)
        model = model.to(device)
        
        optimizer = optim.Adam(
            model.parameters(), lr=conf["learning_rate"], weight_decay=conf["weight_decay"]
        )
        criterion = torch.nn.NLLLoss()
        evaluator = get_evaluator(conf["dataset"])

        """ Data split and run """
        loss_and_score = []
        
        if args.exp_setting == "tran":
            indices = (idx_train, idx_val, idx_test)

            # propagate node feature
            if args.feature_aug_k > 0:
                feats = feature_prop(feats, g, args.feature_aug_k)

            logits, out, score_val, score_test = run_transductive(
                conf,
                model,
                g,
                feats,
                labels,
                indices,
                criterion,
                evaluator,
                optimizer,
                logger,
                loss_and_score,
            )
            score_lst = [score_test]

        elif args.exp_setting == "ind":
            indices = graph_split(idx_train, idx_val, idx_test, args.split_rate, seed)

            # propagate node feature. The propagation for the observed graph only happens within the subgraph obs_g
            if args.feature_aug_k > 0:
                idx_obs = indices[3]
                obs_g = g.subgraph(idx_obs)
                obs_feats = feature_prop(feats[idx_obs], obs_g, args.feature_aug_k)
                feats = feature_prop(feats, g, args.feature_aug_k)
                feats[idx_obs] = obs_feats

            logits, out, score_val, score_test_tran, score_test_ind = run_inductive(
                conf,
                model,
                g,
                feats,
                labels,
                indices,
                criterion,
                evaluator,
                optimizer,
                logger,
                loss_and_score,
            )
            score_lst = [score_test_tran, score_test_ind]
            

        if out_avg is None:
            out_avg = logits
        else:
            out_avg += logits
        
        #获取中间层特征
        g = g.to(device)
        data = g
        #h_list,_ = model.forward_fitnet(data, feats)
        h_list, logits, h_s_proj = model.forward_fitnet(data, feats) 
        if not h_list:
            logger.warning("h_list is empty!")
            # 可以根据实际情况进行处理，例如跳过当前教师模型的中间特征提取
            continue
        hidden_emb = h_list[-1]
        
        # 累加教师模型的特征
        if out_emb is None:
            out_emb = hidden_emb
        else:
            out_emb += hidden_emb
    
    out_avg /= args.num_teacher
    out_avg  = out_avg.log_softmax(dim=1)
    #logger.info(f"Averaged teacher logit is saved.")
    
    out_emb /= args.num_teacher
    #logger.info(f"Averaged teacher emb is saved.")
    


    """ Save averaged teacher logit/emb """
    teacher_names = "_".join(args.teacher)  # 将教师模型名称用下划线连接
    out_avg_np = out_avg.detach().cpu().numpy()
    np.savez(output_dir.joinpath("out"), out_avg_np)
    out_avg = torch.tensor(out_avg_np).to(device)
    
    out_emb_np = out_emb.detach().cpu().numpy()
    np.savez(output_dir.joinpath("emb"), out_emb_np)
    out_emb = torch.tensor(out_emb_np).to(device)
    
    labels = labels.to(device)
    # 根据实验设置选择评分逻辑
    if args.exp_setting == "tran":
        # 对 transductive 任务进行评分
        score_avg = evaluator(out_avg[idx_test], labels[idx_test])
        logger.info(f"Averaged Logit Test Score: {score_avg:.4f}")
        score_lst = [score_avg]
    elif args.exp_setting == "ind":
        # 对 inductive 任务进行评分
        score_avg_tran = evaluator(out_avg[idx_test], labels[idx_test])
        score_avg_ind = evaluator(out_avg[idx_test], labels[idx_test])
        logger.info(f"Averaged Logit Inductive Test Score: {score_avg_ind:.4f}")
        score_lst = [score_avg_tran, score_avg_ind]
        
    #将模型参数打印进日志中
    logger.info(
        f"Teacher model num_layers: {conf['num_layers']}. hidden_dim: {conf['hidden_dim']}. dropout_ratio: {conf['dropout_ratio']}"
    )
    logger.info(f"# params {sum(p.numel() for p in model.parameters())}")


    """ Saving loss curve and model """
    if args.save_results:
        # Loss curves
        loss_and_score = np.array(loss_and_score)
        np.savez(output_dir.joinpath("loss_and_score"), loss_and_score)

        # Model
        torch.save(model.state_dict(), output_dir.joinpath("model.pth"))

    """ Saving min-cut loss """
    if args.exp_setting == "tran" and args.compute_min_cut:
        min_cut = compute_min_cut_loss(g, out)
        with open(output_dir.parent.joinpath("min_cut_loss"), "a+") as f:
            f.write(f"{min_cut :.4f}\n")

    return score_test,score_lst


def repeat_run(args):
    scores = []
    for seed in range(args.num_exp):
        seed = seed
        scores.append(run(args))
    scores_np = np.array(scores)
    return scores_np.mean(axis=0), scores_np.std(axis=0)


def main(num_layers, hidden_dim, dropout_ratio, learning_rate, lamb, seed):
    args = get_args()
    if args.num_exp == 1:
        score, score_lst = run(args, num_layers, hidden_dim, dropout_ratio, learning_rate, lamb, seed)
        score_str = "".join([f"{s : .4f}\t" for s in score_lst])

    elif args.num_exp > 1:
        score_mean, score_std = repeat_run(args)
        score_str = "".join(
            [f"{s : .4f}\t" for s in score_mean] + [f"{s : .4f}\t" for s in score_std]
        )

    with open(args.output_dir.parent.joinpath("exp_results"), "a+") as f:
        f.write(f"{score}\n")

    # for collecting aggregated results
    print(score_str)

    return score #score_str

D={"num_layers":2, "hidden_dim":64, "dropout_ratio":0.5,"learning_rate":0.005, "lamb":1,"seed":0} 
if __name__ == "__main__":
    main(**D)
