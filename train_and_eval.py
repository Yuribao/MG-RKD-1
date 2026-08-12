import numpy as np
import copy
import torch
import dgl
from utils import set_seed
import torch.nn.functional as F
import torch.nn as nn
from sklearn.neighbors import kneighbors_graph
import time
"""
1. Train and eval
"""
class LetKD_Adapter(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        # 查询、键、值的投影层
        self.query_proj = nn.Linear(student_dim, teacher_dim)
        # 教师的嵌入本身就是键和值，不需要投影
        self.student_proj = nn.Linear(student_dim, teacher_dim) 
        
    def forward(self, student_embs, teacher_embs):
        if student_embs.shape[0] == 0:
             return torch.empty(0, teacher_embs.shape[1], device=student_embs.device)

        teacher_dim = teacher_embs.shape[1]
        query = self.query_proj(student_embs)
        
        # 教师的嵌入作为键和值
        key = teacher_embs   
        value = teacher_embs 

        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = attn_scores / (teacher_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
    
        retrieved_knowledge = torch.matmul(attn_probs, value)
        student_embs_projected = self.student_proj(student_embs)
        return retrieved_knowledge, student_embs_projected

def train(model, data, feats, labels, criterion, optimizer, idx_train, lamb):
    """
    GNN full-batch training. Input the entire graph `g` as data.整幅图作为数据输入
    lamb: weight parameter lambda λ
    """
    model.train()

    # Compute loss and prediction
    logits = model(data, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_train], labels[idx_train])
 
    return loss


def train_sage(model, dataloader, feats, labels, criterion, optimizer, lamb):
    """
    Train for GraphSAGE. Process the graph in mini-batches using `dataloader` instead the entire graph `g`.
    大规模图数据,使用迷你批次(mini-batch)进行训练   用于 GraphSAGE 模型的训练，支持小批次处理
    每次迭代时，训练一小批节点和其邻居（由 dataloader 提供）
    lamb: weight parameter lambda
    1遍历数据加载器,获取小批次数据。
    2计算每个小批次的损失并更新模型。
    3返回平均损失
    """
    device = feats.device
    model.train()
    total_loss = 0
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [blk.int().to(device) for blk in blocks]
        batch_feats = feats[input_nodes]
        batch_labels = labels[output_nodes]

        # Compute loss and prediction
        logits = model(blocks, batch_feats)
        out = logits.log_softmax(dim=1)
        loss = criterion(out, batch_labels)
        total_loss += loss.item()

        loss *= lamb
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


def train_mini_batch(model, feats, labels, batch_size, criterion, optimizer, idx_l, lamb):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    在大型数据集上使用小批次训练 MLP 模型。忽略图结构，仅使用节点特征
    lamb: weight parameter lambda
    数据批次化：将节点特征分成小批次进行训练。
    前向传播：对每个批次的节点特征进行前向传播，计算预测结果。
    计算损失：使用损失函数计算当前批次的损失。
    反向传播与优化：计算梯度并通过优化器更新模型。
    
    """
    model.train()

    logits = model(None, feats)
    out = logits.log_softmax(dim=1)
    loss = criterion(out[idx_l], labels[idx_l])
    loss *= lamb
    return loss
    
    
    

def train_mini_batch_temperature(model, data, feats, teacher_outputs, batch_size, criterion, optimizer, temperature, idx_l):
    """
    用温度缩放对教师软标签进行训练。
    对于每个小批次，先将学生的 logits 除以 temperature,再计算 log_softmax
    同时对教师输出logits也进行 temperature 缩放，然后计算 log_softmax
    最后计算 KL 散度，并乘上 temperature^2。
    """
    model.train()

    total_loss = 0
    if "SAGE" in model.model_name: 
        device = feats.device
        for step, (input_nodes, output_nodes, blocks) in enumerate(data):
            blocks = [blk.int().to(device) for blk in blocks]  # 将 blocks 移动到设备
            batch_feats = feats[input_nodes]  # 当前批次的节点特征
            batch_labels = teacher_outputs[output_nodes]  # 教师模型的输出
            
            logit_s = model(blocks, batch_feats) 
            logits_t = batch_labels
            adapter = torch.nn.Linear(logit_s.size(1), logits_t.size(1)).to(device)
            logit_s = adapter(logit_s)

            student_log_prob = torch.log_softmax(logit_s / temperature, dim=1)
            teacher_prob = torch.softmax(logits_t / temperature, dim=1)
            loss = criterion(student_log_prob, teacher_prob) * (temperature ** 2)
            total_loss += loss
  
        return total_loss / len(data)
    else:
        if "MLP" in model.model_name:
    
            num_batches = max(1, feats.shape[0] // batch_size)
            idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]
            if num_batches == 1:
                idx_batch = idx_batch.view(1, -1)
            else:
                idx_batch = idx_batch.view(num_batches, batch_size)
            for i in range(num_batches):    

                logit_s = model(None, feats[idx_batch[i]])

                logits_t = teacher_outputs[idx_batch[i]]
  
                student_log_prob = torch.log_softmax(logit_s / temperature, dim=1)
                teacher_prob = torch.softmax(logits_t / temperature, dim=1)
                loss = criterion(student_log_prob, teacher_prob) * (temperature ** 2)
                
                total_loss += loss
       
            return total_loss  / num_batches 
        else: 
            logit_s = model(data, feats)
   
            logits_t = teacher_outputs
   
            student_log_prob = torch.log_softmax(logit_s / temperature, dim=1)
            teacher_prob = torch.softmax(logits_t / temperature, dim=1)
            loss = (criterion(student_log_prob, teacher_prob) * (temperature ** 2)).mean()
          
            
            loss_val = loss

            return loss_val 
        


def evaluate(model, data, feats, labels, criterion, evaluator, idx_eval=None):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    with torch.no_grad():
        #h_list, _, _ = model.forward_fitnet(data, feats) #Dilidret
        logits = model.inference(data, feats) 
        out = logits.log_softmax(dim=1) #得到每个类别的对数概率soft labels
        if idx_eval is None: 
            loss = criterion(out, labels)
            score = evaluator(out, labels)
        else: #计算指定索引
            loss = criterion(out[idx_eval], labels[idx_eval])
            score = evaluator(out[idx_eval], labels[idx_eval])
    

    return logits, out, loss.item(), score 


def evaluate_mini_batch(
    model, feats, labels, criterion, batch_size, evaluator, idx_eval=None
):
    """
    Evaluate MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    Return:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """

    model.eval()
    with torch.no_grad():
        num_batches = int(np.ceil(len(feats) / batch_size))
        out_list = []
        for i in range(num_batches):
            logits = model.inference(None, feats[batch_size * i : batch_size * (i + 1)])
            out = logits.log_softmax(dim=1)
            out_list += [out.detach()]

        out_all = torch.cat(out_list)

        if idx_eval is None:
            loss = criterion(out_all, labels)
            score = evaluator(out_all, labels)
        else:
            loss = criterion(out_all[idx_eval], labels[idx_eval])
            score = evaluator(out_all[idx_eval], labels[idx_eval])

    return logits, out_all, loss.item(), score


"""
2. Run teacher
"""


def run_transductive(
    conf,
    model,
    g,
    feats,
    labels,
    indices,#训练集、验证集和测试集索引
    criterion,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Train and eval under the transductive setting.
    The train/valid/test split is specified by `indices`.
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    loss_and_score: Stores losses and scores.
    训练和评估模型，其中测试节点是训练图的一部分，通常使用图数据进行训练和评估
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    idx_train, idx_val, idx_test = indices

    feats = feats.to(device)
    labels = labels.to(device)
    
    if "SAGE" in model.model_name:
       
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]

        g = g.to(device)
        data = g
        data_eval = g
    else:
        g = g.to(device)
        data = g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(model, data, feats, labels, criterion, optimizer, lamb)
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats, labels, batch_size, criterion, optimizer, idx_train, lamb)
        else: #GCN GAT SGC
            loss = train(model, data, feats, labels, criterion, optimizer, idx_train, lamb)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                logits, _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                logits, _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                logits, _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion, batch_size, evaluator
                )
            else:
                logits, out, loss_train, score_train = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_train
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])
            print(
                f"Epoch:{epoch:04d} train loss:{loss:.4f} acc:{score_train * 100:.2f} | val loss:{score_val:.4f} acc:{score_test * 100:.2f}"
            )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test,
                    score_train,
                    score_val,
                    score_test,
                ]
            ]

            if score_val > best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:

        logits, out, _, score_val = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_val
        )
    else:
        logits, out, _, score_val = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_val
        )

    score_test = evaluator(out[idx_test], labels[idx_test])
    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )

    return logits, out, score_val, score_test


def run_inductive(
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
):
    """
    Train and eval under the inductive setting.
    The train/valid/test split is specified by `indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large. Thus, SAGE is used for GNNs, mini-batch is used for MLPs.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    loss_and_score: Stores losses and scores.
    在归纳式设置中，训练和评估模型，其中测试节点在训练过程中不可见。通常用于在一个大图中，先训练一个子图（观察图），然后在另一个子图（测试图）上评估模型
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = indices 

    feats = feats.to(device)
    labels = labels.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_g = g.subgraph(idx_obs)

    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        obs_g.create_formats_()
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        obs_dataloader = dgl.dataloading.NodeDataLoader(
            obs_g,
            obs_idx_train,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        obs_dataloader_eval = dgl.dataloading.NodeDataLoader(
            obs_g,
            torch.arange(obs_g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )
        dataloader_eval = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        obs_data = obs_dataloader
        obs_data_eval = obs_dataloader_eval
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        feats_train, labels_train = obs_feats[obs_idx_train], obs_labels[obs_idx_train]
        feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
        feats_test_tran, labels_test_tran = (
            obs_feats[obs_idx_test],
            obs_labels[obs_idx_test],
        )
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    else:
        obs_g = obs_g.to(device)
        g = g.to(device)

        obs_data = obs_g
        obs_data_eval = obs_g
        data_eval = g

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        if "SAGE" in model.model_name:
            loss = train_sage(
                model, obs_data, obs_feats, obs_labels, criterion, optimizer
            )
        elif "MLP" in model.model_name:
            loss = train_mini_batch(
                model, feats_train, labels_train, batch_size, criterion, optimizer
            )
        else:
            loss = train(
                model,
                obs_data,
                obs_feats,
                obs_labels,
                criterion,
                optimizer,
                obs_idx_train,
            )

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                logits, _, loss_train, score_train = evaluate_mini_batch(
                    model, feats_train, labels_train, criterion, batch_size, evaluator
                )
                logits, _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion, batch_size, evaluator
                )
                logits, _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                    model,
                    feats_test_tran,
                    labels_test_tran,
                    criterion,
                    batch_size,
                    evaluator,
                )
                logits, _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                    model,
                    feats_test_ind,
                    labels_test_ind,
                    criterion,
                    batch_size,
                    evaluator,
                )
            else:
                logits, obs_out, loss_train, score_train = evaluate(
                    model,
                    obs_data_eval,
                    obs_feats,
                    obs_labels,
                    criterion,
                    evaluator,
                    obs_idx_train,
                )
                # Use criterion & evaluator instead of evaluate to avoid redundant forward pass
                loss_val = criterion(
                    obs_out[obs_idx_val], obs_labels[obs_idx_val]
                ).item()
                score_val = evaluator(obs_out[obs_idx_val], obs_labels[obs_idx_val])
                loss_test_tran = criterion(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                ).item()
                score_test_tran = evaluator(
                    obs_out[obs_idx_test], obs_labels[obs_idx_test]
                )

                # Evaluate the inductive part with the full graph
                logits, out, loss_test_ind, score_test_ind = evaluate(
                    model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_train: {score_train:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_train,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_train,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]
            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    if "MLP" in model.model_name:
        logits, obs_out, _, score_val = evaluate_mini_batch(
            model, obs_feats, obs_labels, criterion, batch_size, evaluator, obs_idx_val
        )
        logits, out, _, score_test_ind = evaluate_mini_batch(
            model, feats, labels, criterion, batch_size, evaluator, idx_test_ind
        )

    else:
        _, obs_out, _, score_val = evaluate(
            model,
            obs_data_eval,
            obs_feats,
            obs_labels,
            criterion,
            evaluator,
            obs_idx_val,
        )
        logits, out, _, score_test_ind = evaluate(
            model, data_eval, feats, labels, criterion, evaluator, idx_test_ind
        )

    score_test_tran = evaluator(obs_out[obs_idx_test], obs_labels[obs_idx_test])
    out[idx_obs] = obs_out
    logger.info(
        f"Best valid model at epoch: {best_epoch :3d}, score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return logits, out, score_val, score_test_tran, score_test_ind


"""
3. Distill
"""


def distill_run_transductive(
    conf,
    model,
    g, # 图结构数据（对于 MLP 为 None，对于 GNN 为 `g` 或 `blocks`）
    feats,
    labels,
    teacher_num_layers,
    out_t_all, # 教师模型软标签
    emb_t_all, # 教师模型特征
    distill_indices,  # 数据划分索引，包括硬标签训练集、软标签训练集、验证集和测试集的索引
    criterion_l, # 硬标签损失函数
    criterion_t, # 软标签损失函数
    evaluator,
    optimizer,
    logger,
    loss_and_score
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.教师模型生成的软标签
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    teacher_num_layers = conf["teacher_num_layers"]
    lamb = conf["lamb"]
    lamb2 = conf["lamb2"]
    weight = conf["weight"]
    temperature = conf["temperature"] #引入温度
    idx_l, idx_t, idx_val, idx_test = distill_indices 
    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    emb_t_all = emb_t_all.to(device)

    src, dst = g.edges()
    src = src.to(device)
    dst = dst.to(device)
    
    # 先获取学生模型输出特征维度（通过一次前向传播）
    with torch.no_grad():
        if "MLP" not in model.model_name:
            dummy_input_g = g.to(device) 
        else:
            dummy_input_g = None
            

        dummy_s_h_list, _, _ = model.forward_fitnet(dummy_input_g, feats)
        num_layers = len(dummy_s_h_list)
        distill_layer_idx = num_layers // 2
        student_feat_dim = dummy_s_h_list[distill_layer_idx].size(1)
        teacher_feat_dim = emb_t_all.size(1)

    letkd_adapter = LetKD_Adapter(student_feat_dim, teacher_feat_dim).to(device)
    optimizer_letkd_adapter = torch.optim.Adam(letkd_adapter.parameters(), lr=conf["learning_rate"],weight_decay=conf["weight_decay"])
    adapter = torch.nn.Linear(student_feat_dim, teacher_feat_dim).to(device)

    optimizer_adapter = torch.optim.Adam(
        adapter.parameters(), 
        lr=conf["learning_rate"], 
        weight_decay=conf["weight_decay"]
    )


    if "SAGE" in model.model_name:
        # Create dataloader for SAGE

        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [eval(fanout) for fanout in conf["fan_out"].split(",")]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            idx_t,
            sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        # SAGE inference is implemented as layer by layer, so the full-neighbor sampler only collects one-hop neighors
        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader_eval = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=conf["num_workers"],
        )

        data = dataloader
        data_eval = dataloader_eval
    elif "MLP" in model.model_name:
        g = g.to(device)
        data = g
        feats_l, labels_l = feats[idx_l], labels[idx_l]
        feats_t, out_t = feats[idx_t], out_t_all[idx_t] #全部数据
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]
    else:#GNN
        g = g.to(device)
        data = g
        data_eval = g
    
    """
    feats_l, labels_l = feats[idx_l], labels[idx_l] 
    feats_t, out_t = feats[idx_t], out_t_all[idx_t] 
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]
    """
    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        adapter.train()
        letkd_adapter.train()


        # 硬标签损失
        if "MLP" in model.model_name:
            loss_l = train_mini_batch(
                model, feats, labels, batch_size, criterion_l, optimizer, idx_l, lamb
            )
            
            loss_t = train_mini_batch_temperature(
                model, g,  feats , out_t_all, batch_size, criterion_t, optimizer, temperature, idx_l
            )
        
        elif "SAGE" in model.model_name:
            loss_l = train_sage(
                model, data, feats, labels, criterion_l, optimizer, lamb
            )
            loss_t = train_mini_batch_temperature(
                model, data, feats, out_t_all, batch_size, criterion_t, optimizer, temperature, idx_l
            )
        
        else: #GCN
            loss_l = train(
                model, data, feats, labels, criterion_l, optimizer, idx_l, lamb
            )
            
            loss_t = train_mini_batch_temperature(
                model, data, feats, out_t_all, batch_size, criterion_t, optimizer, temperature, idx_l
            )
            
 

        s_h_list, logits_s, h_s_proj = model.forward_fitnet(data, feats)
        
        emb_s_intermediate = s_h_list[distill_layer_idx]
        emb_t_intermediate = emb_t_all  
        
        
        batch_size_kd = 1024
        num_nodes = emb_s_intermediate.shape[0]

        if num_nodes > batch_size_kd:
            perm = torch.randperm(num_nodes, device=device)[:batch_size_kd]
            batch_emb_s = emb_s_intermediate[perm] # [B, D_s]
            batch_emb_t = emb_t_intermediate[perm] # [B, D_t]
        else:
            batch_emb_s = emb_s_intermediate
            batch_emb_t = emb_t_intermediate
        retrieved_knowledge, student_embs_projected = letkd_adapter(batch_emb_s, batch_emb_t)

        loss_letkd = F.mse_loss(student_embs_projected, retrieved_knowledge.detach())


        if h_s_proj.shape[1] != emb_t_all.shape[1]:
             # 如果没有，可以用 LetKD 的 projected output
             h_s_aligned = letkd_adapter.student_proj(h_s_proj)
        else:
             h_s_aligned = h_s_proj
        
        emb_s_adapted = adapter(emb_s_intermediate)


        if data is not None and g.num_edges() > 0:
                src, dst = g.edges()
                src = src.to(device)
                dst = dst.to(device)

                h_s = h_s_proj
                h_t = emb_t_all

                h_t_norm = F.normalize(h_t, p=2, dim=1).detach() 
                h_s_norm = F.normalize(h_s, p=2, dim=1)


                total_struct_loss = 0
                num_edges = src.shape[0]
                struct_batch_size = 20000 
                for i in range(0, num_edges, struct_batch_size):
                    end = min(i + struct_batch_size, num_edges)
                    batch_src = src[i:end]
                    batch_dst = dst[i:end]

                    t_src_feat = h_t_norm[batch_src]
                    t_dst_feat = h_t_norm[batch_dst]
                    sim_t = torch.sum(t_src_feat * t_dst_feat, dim=1)

                    s_src_feat = h_s_norm[batch_src]
                    s_dst_feat = h_s_norm[batch_dst]
                    sim_s = torch.sum(s_src_feat * s_dst_feat, dim=1)
                    edge_weights = 1 - torch.sigmoid(sim_t / temperature)
                    batch_loss = (edge_weights * (sim_s - sim_t) ** 2).sum()
                    total_struct_loss += batch_loss

                loss_structure = total_struct_loss / num_edges

        else:
            loss_structure = torch.tensor(0.0, device=device)
            


   
        distill_loss = lamb * loss_t + weight * loss_letkd  + lamb2 * loss_structure 
        loss =  0.1* loss_l + 0.9* distill_loss
 


        optimizer.zero_grad()
        optimizer_adapter.zero_grad()
        optimizer_letkd_adapter.zero_grad()

        loss.backward()
        optimizer.step()
        optimizer_adapter.step()
        optimizer_letkd_adapter.step()
        

        if epoch % conf["eval_interval"] == 0:
            if "MLP" in model.model_name:
                logits, _, loss_l, score_l = evaluate_mini_batch(
                    model, feats_l, labels_l, criterion_l, batch_size, evaluator
                )
                logits, _, loss_val, score_val = evaluate_mini_batch(
                    model, feats_val, labels_val, criterion_l, batch_size, evaluator
                )
                logits, _, loss_test, score_test = evaluate_mini_batch(
                    model, feats_test, labels_test, criterion_l, batch_size, evaluator
                )
                print(
                    f"Epoch:{epoch:04d} train loss:{loss:.4f} acc:{score_l * 100:.2f} | val loss:{loss_val:.4f} acc:{score_val * 100:.2f}"
                )
            else:
                _, out, loss_l, score_l = evaluate(
                    model, data_eval, feats, labels, criterion_l, evaluator, None #idx_val
                )
                loss_val = criterion_l(out[idx_val], labels[idx_val]).item()
                score_val = evaluator(out[idx_val], labels[idx_val])
                loss_test = criterion_l(out[idx_test], labels[idx_test]).item()
                score_test = evaluator(out[idx_test], labels[idx_test])
                print(
                    f"Epoch:{epoch:04d} train loss:{loss:.4f} acc:{score_l * 100:.2f} | val loss:{loss_val:.4f} acc:{score_val * 100:.2f}"
                )
            logger.debug(
                f"Ep {epoch:3d} | loss: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_test: {score_test:.4f}"
            )
            loss_and_score += [
                [epoch, loss_l, loss_val, loss_test, score_l, score_val, score_test]
            ]

            if score_val > best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = {
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'adapter_state_dict': copy.deepcopy(letkd_adapter.state_dict()),
                'adapter_state_dict1': copy.deepcopy(adapter.state_dict()),
                }
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    #model.load_state_dict(state)
    model.load_state_dict(state['model_state_dict'])
    letkd_adapter.load_state_dict(state['adapter_state_dict'])
    adapter.load_state_dict(state['adapter_state_dict1'])
    #adaptive_temp_net.load_state_dict(state['adapter_state_dict2'])
    if "MLP" in model.model_name:
        logits, out, _, score_val = evaluate_mini_batch(
            model, feats, labels, criterion_l, batch_size, evaluator, idx_val
        )
    else:
        _, out, _, score_val = evaluate(
            model, data_eval, feats, labels, criterion_l, evaluator, idx_val
        )
    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test = evaluator(out[idx_test], labels[idx_test])

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d}, score_val: {score_val :.4f}, score_test: {score_test :.4f}"
    )
    
    print(f"\n--- Starting t-SNE for Distilled Student on {conf['dataset']} ---")
    model.eval()
    with torch.no_grad():
        s_h_list, final_logits, _ = model.forward_fitnet(g.to(device), feats.to(device))
        
        # 取最后一层特征
        X_to_plot = final_logits.cpu().numpy()
        student_emb = s_h_list[-1].cpu().numpy() 
        y_labels = labels.cpu().numpy()
        tsne_save_path = f"tsne_student_{conf['dataset']}.png" 

    
    return out, score_val, score_test


def distill_run_inductive(
    conf,
    model,
    feats,
    labels,
    out_t_all,
    distill_indices,
    criterion_l,
    criterion_t,
    evaluator,
    optimizer,
    logger,
    loss_and_score,
):
    """
    Distill training and eval under the inductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    idx starting with `obs_idx_` contains the node idx in the observed graph `obs_g`.
    idx starting with `idx_` contains the node idx in the original graph `g`.
    The model is trained on the observed graph `obs_g`, and evaluated on both the observed test nodes (`obs_idx_test`) and inductive test nodes (`idx_test_ind`).
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    idx_obs: Idx of nodes in the original graph `g`, which form the observed graph 'obs_g'.
    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively.
    loss_and_score: Stores losses and scores.
    """

    set_seed(conf["seed"])
    device = conf["device"]
    batch_size = conf["batch_size"]
    lamb = conf["lamb"]
    (
        obs_idx_l,
        obs_idx_t,
        obs_idx_val,
        obs_idx_test,
        idx_obs,
        idx_test_ind,
    ) = distill_indices

    feats = feats.to(device)
    labels = labels.to(device)
    out_t_all = out_t_all.to(device)
    obs_feats = feats[idx_obs]
    obs_labels = labels[idx_obs]
    obs_out_t = out_t_all[idx_obs]

    feats_l, labels_l = obs_feats[obs_idx_l], obs_labels[obs_idx_l]
    feats_t, out_t = obs_feats[obs_idx_t], obs_out_t[obs_idx_t]
    feats_val, labels_val = obs_feats[obs_idx_val], obs_labels[obs_idx_val]
    feats_test_tran, labels_test_tran = (
        obs_feats[obs_idx_test],
        obs_labels[obs_idx_test],
    )
    feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    best_epoch, best_score_val, count = 0, 0, 0
    for epoch in range(1, conf["max_epoch"] + 1):
        loss_l = train_mini_batch(
            model, feats_l, labels_l, batch_size, criterion_l, optimizer, lamb
        )
        loss_t = train_mini_batch(
            model, feats_t, out_t, batch_size, criterion_t, optimizer, lamb
        )
        loss = loss_l + loss_t
        if epoch % conf["eval_interval"] == 0:
            logits, _, loss_l, score_l = evaluate_mini_batch(
                model, feats_l, labels_l, criterion_l, batch_size, evaluator
            )
            logits, _, loss_val, score_val = evaluate_mini_batch(
                model, feats_val, labels_val, criterion_l, batch_size, evaluator
            )
            logits, _, loss_test_tran, score_test_tran = evaluate_mini_batch(
                model,
                feats_test_tran,
                labels_test_tran,
                criterion_l,
                batch_size,
                evaluator,
            )
            logits, _, loss_test_ind, score_test_ind = evaluate_mini_batch(
                model,
                feats_test_ind,
                labels_test_ind,
                criterion_l,
                batch_size,
                evaluator,
            )

            logger.debug(
                f"Ep {epoch:3d} | l: {loss:.4f} | s_l: {score_l:.4f} | s_val: {score_val:.4f} | s_tt: {score_test_tran:.4f} | s_ti: {score_test_ind:.4f}"
            )
            loss_and_score += [
                [
                    epoch,
                    loss_l,
                    loss_val,
                    loss_test_tran,
                    loss_test_ind,
                    score_l,
                    score_val,
                    score_test_tran,
                    score_test_ind,
                ]
            ]

            if score_val >= best_score_val:
                best_epoch = epoch
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

        if count == conf["patience"] or epoch == conf["max_epoch"]:
            break

    model.load_state_dict(state)
    logits, obs_out, _, score_val = evaluate_mini_batch(
        model, obs_feats, obs_labels, criterion_l, batch_size, evaluator, obs_idx_val
    )
    logits, out, _, score_test_ind = evaluate_mini_batch(
        model, feats, labels, criterion_l, batch_size, evaluator, idx_test_ind
    )

    # Use evaluator instead of evaluate to avoid redundant forward pass
    score_test_tran = evaluator(obs_out[obs_idx_test], labels_test_tran)
    out[idx_obs] = obs_out

    logger.info(
        f"Best valid model at epoch: {best_epoch: 3d} score_val: {score_val :.4f}, score_test_tran: {score_test_tran :.4f}, score_test_ind: {score_test_ind :.4f}"
    )
    return out, score_val, score_test_tran, score_test_ind
