import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GraphAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(GraphAttentionLayer, self).__init__()
        self.gat = GATConv(d_model, d_model // nhead, heads=nhead, concat=True)

    def forward(self, x, edge_index):

        out,atten_score = self.gat(x, edge_index,return_attention_weights=True)# 假设你已经获取了atten_score，并且它的形状为[Num_edges, Num_heads]
        #atten_score 有两个，一个是attenweight，一个是edge

        return out, atten_score
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn= nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.graph_attn = GraphAttentionLayer(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model + 1, d_model * 4),  # +1 for condition
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, condition=None):
        # 自注意力
        attn_output, self.score  = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        # 图注意力
        x_reshaped = x.transpose(0, 1).reshape(-1, x.size(-1))
        graph_output,self.graph_score = self.graph_attn(x_reshaped, edge_index)
        graph_output = graph_output.reshape(x.size(1), x.size(0), -1).transpose(0, 1)
        x = self.norm2(x + graph_output)
        
        # 前馈网络（加入condition）
        if condition is not None:
            # 确保 condition 的维度与 x 匹配
            condition = condition[:x.size(0), :, :]  # Truncate or pad as necessary
            x_with_condition = torch.cat([x, condition], dim=-1)
        else:
            # 如果没有 condition，用零填充
            zero_condition = torch.zeros(x.size(0), x.size(1), 1, device=x.device) #[2500,4,1]
            
            x_with_condition = torch.cat([x, zero_condition], dim=-1)
        
        ffn_output = self.ffn(x_with_condition)
        x = self.norm3(x + ffn_output)
        
        return x
    def _get_attention(self):
        print(self.score.shape) #torch.Size([2500, 32, 20])
        return self.score
    def _get_graph_attention(self):
        print(len(self.graph_score)) #
        return self.graph_score

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, num_sensors):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, output_dim)
        self.num_sensors = num_sensors

    def forward(self, x, adjacency_matrix, condition=None):
        '''
        x: [batch_size, seq_len, feature_dim=3]
        condition: [batch_size, seq_len, 1] or None
        '''
        x = x.transpose(0, 1)  # [seq_len, batch_size, feature_dim]
        x = self.embedding(x)

        # Create edge_index from adjacency matrix
        edge_index = adjacency_matrix.nonzero().t().contiguous()

        for layer in self.layers:
            x = layer(x, edge_index, condition)

        x = self.output_layer(x)
        return x.transpose(0, 1)  # [batch_size, seq_len, output_dim]
    def _get_layer_attention(self):
        return self.layers[1]._get_attention()
    def _get_graph_attention(self):
        return self.layers[1]._get_graph_attention()
# 创建 DataLoader

import numpy as np
def calculate_distance_matrix(coords):
    num_points = coords.shape[0]
    distance_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(num_points):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distance_matrix
def return_adj_matrix():
    # 欧几里得坐标
    '(x,y)'
    coordinates = np.array([
        [0,1],
        [-0.1, 0.8], 
        [-0.1, 0.6], [-0.2, 0.6],[-0.3,0.6],
        [-0.1,0.4],[-0.2,0.4],[-0.3,0.4],[-0.4,0.4],[-0.5,0.4],
        [-0.1,0.2],[-0.3,0.2],[-0.4,0.2],[-0.5,0.2],[-0.6,0.2],[-0.7,0.2]]#
    )
    epsilon =1e-5

    distance_matrix = calculate_distance_matrix(coordinates)

    # Convert distance matrix to adjacency matrix with inverse weights and add epsilon
    adj_matrix = np.zeros_like(distance_matrix)
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if i != j and distance_matrix[i, j] != 0:
                adj_matrix[i, j] = 1 / (distance_matrix[i, j] + epsilon)
    # Row-normalize the adjacency matrix
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    adj_matrix_normalized = adj_matrix / row_sums

    
    return torch.from_numpy(adj_matrix_normalized)

# 创建模拟邻接矩阵（在实际应用中，这应该根据传感器的实际位置来定义）
# 创建训练集
if __name__ =="__main__":
    # 创建模型实例
    num_sensors =16
    model = TransformerDecoder(
    input_dim=16,
    d_model=20,
    nhead=4,
    num_layers=4,
    output_dim=1, #反问题角度
    num_sensors=num_sensors
    ).to(device)
