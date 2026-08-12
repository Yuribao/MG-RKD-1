import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, SAGEConv, APPNPConv, GATConv, SGConv
#https://www.less-bug.com/posts/gcn-basis-graphconv-gatconv-sageconv-implementation-pyg-dgl/
#https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1.0):
        """
        mode: 'PN' (PairNorm) - 保持成对距离恒定
               'PN-SI' (Scale-Invariant) - 不重新缩放，只去中心化
        scale: 缩放因子 s
        """
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        # x: [N, D]
        col_mean = x.mean(dim=0, keepdim=True)      
        if self.mode == 'PN':
            # 1. 去中心化 (Centering)
            x = x - col_mean
            # 2. 重新缩放 (Rescaling)
            # 计算行范数 (Row Norm) 的平均值
            row_norm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            # 强制拉伸特征，使其总能量保持不变
            x = self.scale * x / row_norm_mean
            
        return x
    
class WeightedResGCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, norm_type, use_residual=True):
        super(WeightedResGCNBlock, self).__init__()
        # DGL 的 GraphConv
        self.conv = GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm_type
        self.use_residual = use_residual

        # 归一化层选择
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm_type == "pair": 
            # 【关键点】在这里初始化 PairNorm
            # scale=1.0 是默认值，有时深层网络设为 10.0 效果更好，可作为超参调整
            self.norm = PairNorm(scale=10.0) 
        else:
            self.norm = None

        # --- Paper 3 核心：稀疏聚合系数 Xi ---
        # 只有当维度一致且启用残差时，才初始化这个参数
        if self.use_residual and in_dim == out_dim:
            # 初始化为 0.5，让模型自己学
            self.xi = nn.Parameter(torch.tensor(0.5)) 
        else:
            # 如果不需要残差（如维度变化），注册一个 None 或者是固定值
            self.register_parameter('xi', None)

    def forward(self, g, h_in):
        # 1. 卷积
        h_conv = self.conv(g, h_in)
        
        # 2. 归一化 (PairNorm 在这里起作用)
        # 它的作用是把卷积后变平滑的特征重新“撑开”
        if self.norm is not None:
            h_conv = self.norm(h_conv)
        
        # 3. 激活函数
        if self.activation is not None:
            h_conv = self.activation(h_conv)
            
        # 4. Dropout
        h_conv = self.dropout(h_conv)
        
        # 5. --- Paper 3 实现：加权残差 ---
        # 公式：H(l+1) = alpha * Conv(H) + (1 - alpha) * H_old
        if self.use_residual and self.xi is not None and h_conv.shape == h_in.shape:
            alpha = torch.sigmoid(self.xi) 
            h_out = alpha * h_conv + (1 - alpha) * h_in
        else:
            h_out = h_conv
            
        return h_out

    
class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none", #"batch"（批归一化）、"layer"（层归一化）或 "none"（无归一化）
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList() #用于存储神经网络中的线性层 self.layers 被初始化为一个空的 ModuleList，然后通过 self.layers.append() 方法添加了多个 nn.Linear 层
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim)) # 添加批归一化层 列归一化
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim)) # 添加层归一化层 行归一化

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats #输入特征
        h_list = [] #隐藏层输出列表
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list, h


"""
Adapted from the SAGE implementation from the official DGL example
https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py
"""


class SAGE(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList() #用于存储 GraphSAGE 的卷积层
        self.norms = nn.ModuleList() #用于存储归一化层

        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, "gcn")) #GCN聚合方案
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, "gcn"))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, "gcn"))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(SAGEConv(hidden_dim, output_dim, "gcn"))

    def forward(self, blocks, feats):
        h = feats
        h_list = []
        for l, (layer, block) in enumerate(zip(self.layers, blocks)): #遍历每一层的SAGEConv层和对应的Block对象
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            #在第一次迭代时，h 初始化为 feats；在后续迭代中，h 是上一层处理后的节点特征
            h_dst = h[: block.num_dst_nodes()]#从输入特征h中提取目标节点（RHS）的特征
            # Then we compute the updated representation on the RHS.
            # The shape of h = (num_nodes_RHS, D)
            #源节点（邻居节点）和目标节点（当前节点），需要同时访问源节点和当前目标节点的特征，以便进行邻居信息的聚合和目标节点特征的更新
            h = layer(block, (h, h_dst)) #调用当前的SAGEConv层，对输入特征进行卷积操作，更新目标节点的特征
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h_list, h

    def inference(self, dataloader, feats):
        """
        使用GraphSAGE模型对全邻居进行推理(即没有邻居采样)。
        dataloader:整个图以块的形式加载，每个节点都有完整的邻居。
        feats:整个节点集的输入feats
        """
        device = feats.device
        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                feats.shape[0],
                self.hidden_dim if l != self.num_layers - 1 else self.output_dim,
            ).to(device)
            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = feats[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != self.num_layers - 1:
                    if self.norm_type != "none":
                        h = self.norms[l](h)
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            feats = y
        return y

#GCN + Initial connection
class GCN(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim, #节点特征维度
        hidden_dim, #隐藏层维度
        output_dim, #输出类别数
        dropout_ratio, #防止过拟合
        activation, #激活函数
        norm_type="none", #归一化类别 
        residual=True,
        # alpha=0.2, 
        # beta=0.5
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        # self.alpha = alpha 
        # self.beta = beta
        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))
            # elif self.norm_type == "pair":  
            #     self.norms.append(PairNorm())

            for i in range(num_layers - 2):
                self.layers.append(
                    GraphConv(hidden_dim, hidden_dim, activation=activation)
                )
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))
                # elif self.norm_type == "pair":  
                #     self.norms.append(PairNorm())
            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        h_list = []
        h0 = None 
        for l, layer in enumerate(self.layers): #layer 是 self.layers 列表中的元素，表示当前遍历的神经网络层
            #h_in = h#
            h = layer(g, h)
            if l != self.num_layers - 1:
                #h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h) # 表示对第l层输入 h 进行归一化操作
                h = self.dropout(h)
                if l == 0:
                    h0 = h.clone()
                else:
                    if self.residual: 
                        h = 0.3*h + 0.7*h0 
                    h_list.append(h)
        return h_list, h #h是最终节点特征 h_list是所有层节点列表

# GCN
# class GCN(nn.Module):
#     def __init__(
#         self,
#         num_layers,
#         input_dim, #节点特征维度
#         hidden_dim, #隐藏层维度
#         output_dim, #输出类别数
#         dropout_ratio, #防止过拟合
#         activation, #激活函数
#         norm_type="none", #归一化类别 
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.norm_type = norm_type
#         self.activation = activation
#         self.dropout = nn.Dropout(dropout_ratio)
#         self.layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         if num_layers == 1:
#             self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
#         else:
#             self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
#             if self.norm_type == "batch":
#                 self.norms.append(nn.BatchNorm1d(hidden_dim))
#             elif self.norm_type == "layer":
#                 self.norms.append(nn.LayerNorm(hidden_dim))

#             for i in range(num_layers - 2):
#                 self.layers.append(
#                     GraphConv(hidden_dim, hidden_dim, activation=activation)
#                 )
#                 if self.norm_type == "batch":
#                     self.norms.append(nn.BatchNorm1d(hidden_dim))
#                 elif self.norm_type == "layer":
#                     self.norms.append(nn.LayerNorm(hidden_dim))
#             self.layers.append(GraphConv(hidden_dim, output_dim))


#     def forward(self, g, feats):
#         h = feats
#         h_list = []
#         for l, layer in enumerate(self.layers): #layer 是 self.layers 列表中的元素，表示当前遍历的神经网络层
#             h = layer(g, h)
#             if l != self.num_layers - 1:
#                 h_list.append(h)
#                 if self.norm_type != "none":
#                     h = self.norms[l](h) # 表示对第l层输入 h 进行归一化操作
#                 h = self.dropout(h)
#         return h_list, h #h是最终节点特征 h_list是所有层节点列表


# GAT + Initial connection
class GAT(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        num_heads=8,
        attn_drop=0.3, 
        negative_slope=0.2, 
        residual=False, # GATConv 内部的 ResNet (上一层残差)
        use_initial_residual=True, # 是否开启 h0 初始残差
        alpha=0.7# h0 的保留比例
    ):
        super(GAT, self).__init__()
        assert num_layers > 1

        self.total_hidden_dim = hidden_dim 
        head_dim = hidden_dim // num_heads 
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.use_initial_residual = use_initial_residual
        self.alpha = alpha
        
        if self.use_initial_residual:
            self.h0_proj = nn.Linear(input_dim, head_dim * num_heads)
        
        heads = ([num_heads] * num_layers) + [1]
        
        self.layers.append(
            GATConv(
                input_dim,
                head_dim, # 输出每个头的维度
                heads[0],
                dropout_ratio,
                attn_drop,
                negative_slope, 
                False, # 第一层通常不加内部 residual
                self.activation,
            )
        )

        # Middle Layers
        for l in range(1, num_layers - 1):
            self.layers.append(
                GATConv(
                    head_dim * heads[l - 1], 
                    head_dim,
                    heads[l],
                    dropout_ratio,
                    attn_drop,
                    negative_slope,
                    residual, # 内部 residual
                    self.activation,
                )
            )

        # Output Layer
        self.layers.append(
            GATConv(
                head_dim * heads[-2],
                output_dim,
                heads[-1],
                dropout_ratio,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, feats):
        h = feats
        h_list = []
        h0 = None
        
        if self.use_initial_residual:
            h0 = self.h0_proj(feats)
            if self.activation is not None:
                h0 = self.activation(h0)

        for l, layer in enumerate(self.layers):
            h = layer(g, h)

            if l != self.num_layers - 1:
                h = h.flatten(1) 
    
                if self.use_initial_residual and h0 is not None:
                    if h.shape == h0.shape:
                        h = (1 - self.alpha) * h + self.alpha * h0
                
                h_list.append(h)
            
            else:
                h = h.mean(1)
                h_list.append(h)
                
        return h_list, h



# GAT
# class GAT(nn.Module):
#     def __init__(
#         self,
#         num_layers,
#         input_dim,
#         hidden_dim,
#         output_dim,
#         dropout_ratio,
#         activation,
#         num_heads=8,
#         attn_drop=0.3, #Droupout
#         negative_slope=0.2, # LeakyReLU 负斜率
#         residual=False,
#     ):
#         super(GAT, self).__init__()
#         # For GAT, the number of layers is required to be > 1
#         assert num_layers > 1

#         hidden_dim //= num_heads #整除 多头注意力机制会将特征维度分配到每个头
#         self.num_layers = num_layers
#         self.layers = nn.ModuleList()
#         self.activation = activation

#         heads = ([num_heads] * num_layers) + [1]
#         # input (no residual)
#         self.layers.append(
#             GATConv(
#                 input_dim,
#                 hidden_dim,
#                 heads[0], #头数
#                 dropout_ratio,
#                 attn_drop,
#                 negative_slope, 
#                 False,
#                 self.activation,
#             )
#         )

#         for l in range(1, num_layers - 1):
#             # due to multi-head, the in_dim = hidden_dim * num_heads
#             self.layers.append(
#                 GATConv(
#                     hidden_dim * heads[l - 1], #hidden_dim * heads[l-1]
#                     hidden_dim,
#                     heads[l],
#                     dropout_ratio,
#                     attn_drop,
#                     negative_slope,
#                     residual,
#                     self.activation,
#                 )
#             )

#         self.layers.append(
#             GATConv(
#                 hidden_dim * heads[-2],
#                 output_dim,
#                 heads[-1],
#                 dropout_ratio,
#                 attn_drop,
#                 negative_slope,
#                 residual,
#                 None,
#             )
#         )

#     def forward(self, g, feats):
#         h = feats
#         h_list = []
#         for l, layer in enumerate(self.layers):
#             # [num_head, node_num, nclass] -> [num_head, node_num*nclass]
#             h = layer(g, h)
#             if l != self.num_layers - 1:
#                 h = h.flatten(1) #将输出特征 h 展平（flatten(1)）并添加到 h_list 中
#                 h_list.append(h)
#             else:
#                 h = h.mean(1)
#         return h_list, h



class SGC(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
        K = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SGConv(input_dim, output_dim, k=K, bias=True))
        else:
            self.layers.append(SGConv(input_dim, hidden_dim, k=K, bias=True))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(SGConv(hidden_dim, hidden_dim, k=K, bias=True))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(SGConv(hidden_dim, output_dim, k=K, bias=True))

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h_list,h

    

class APPNP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        activation,
        norm_type="none",
        edge_drop=0.5,
        alpha=0.5,
        k=10,
    ):

        super(APPNP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        h = feats
        h_list = []
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                h_list.append(h)
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = self.activation(h)
                h = self.dropout(h)

        h = self.propagate(g, h)
        return h_list, h



class JKNet(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, norm_type="none"):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_ratio)


        self.layers.append(GraphConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim))
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))


        self.classifier = nn.Linear(num_layers * hidden_dim, output_dim)

    def forward(self, g, feats):
        h = feats
        h_list = [] 
        
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.norms):
                h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            h_list.append(h) # 记录每一层
            

        h_jump = torch.cat(h_list, dim=-1)
        logits = self.classifier(h_jump)
        
        return h_list, logits    
    
    
    
class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()
        self.model_name = conf["model_name"]
        self.num_layers = conf["num_layers"]
        self.hidden_dim = conf["hidden_dim"]

        similarity_dim = conf.get("similarity_dim", self.hidden_dim) 
        # self.hidden_dim 是中间隐藏层的维度（例如 128 或 512）
        self.similarity_proj = nn.Linear(self.hidden_dim, similarity_dim)
        if "MLP" in conf["model_name"]:
            self.encoder = MLP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "SAGE" in conf["model_name"]:
            self.encoder = SAGE(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "GCN" in conf["model_name"]:
            self.encoder = GCN(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],

                
            ).to(conf["device"])
        elif "GAT" in conf["model_name"]:
            self.encoder = GAT(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                attn_drop=conf["attn_dropout_ratio"],
            ).to(conf["device"])
        elif "SGC" in conf["model_name"]:
            self.encoder = SGC(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "APPNP" in conf["model_name"]:
            self.encoder = APPNP(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"],
            ).to(conf["device"])
        elif "JKNet" in conf["model_name"]:
            self.encoder = JKNet(
                num_layers=conf["num_layers"],
                input_dim=conf["feat_dim"],
                hidden_dim=conf["hidden_dim"],
                output_dim=conf["label_dim"],
                dropout_ratio=conf["dropout_ratio"],
                activation=F.relu,
                norm_type=conf["norm_type"]
            ).to(conf["device"])


    def forward(self, data, feats):
        """
        data: a graph `g` or a `dataloader` of blocks
        """
        if "MLP" in self.model_name:
            return self.encoder(feats)[1] #最终输出
        else:
            return self.encoder(data, feats)[1] #预测结果

        
    def forward_fitnet(self, data, feats): #返回的是中间层和最终输出。适用于需要获取隐藏层输出的情况

        if "MLP" in self.model_name:
            #return self.encoder(feats)
            h_list, h = self.encoder(feats)
        else:
            #return self.encoder(data, feats)
            h_list, h = self.encoder(data, feats)
        if self.num_layers == 1:
            h_intermediate = h_list[-1] if h_list else feats 
        else:
            distill_layer_idx = self.num_layers // 2 - 1 
            h_intermediate = h_list[distill_layer_idx]

        if h_intermediate.shape[0] > 0:
            h_s_proj = self.similarity_proj(h_intermediate)
        else:
            h_s_proj = torch.empty(0, self.similarity_proj.out_features, device=h_intermediate.device)

        return h_list, h, h_s_proj 

    def inference(self, data, feats):
        if "SAGE" in self.model_name:
            return self.encoder.inference(data, feats)
        else:
            return self.forward(data, feats)
