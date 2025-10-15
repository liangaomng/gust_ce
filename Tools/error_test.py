
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_rmse(predictions, targets):
    # 计算均方误差（MSE）
    mse = F.mse_loss(predictions, targets)+1e-6
    # 开方得到RMSE
    rmse = torch.sqrt(mse)
    return rmse

def count_parameters(model):
    # 计算模型的总参数数量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params