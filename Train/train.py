import torch
import matplotlib.pyplot as plt
import wandb
import shutil
import os
import torch.nn as nn
import logging
from datetime import datetime
import yaml
import argparse
import torch.optim as optim
from typing import Optional, Dict
import torch.optim as optim
import warnings
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
warnings.filterwarnings('ignore')
import sys
print("Current Working Directory:", os.getcwd())
# Add the directory to the sys.path
directory_path = os.path.dirname(os.getcwd())
# Optionally, print all paths to confirm the addition
for path in sys.path:
    print("Import path:", path)
sys.path.append(directory_path)
from Gust.Tools.error_test import calculate_rmse
import functools
import logging
import wandb
import torch.optim as optim
from tqdm import tqdm
import random
import gc
import numpy as np
from Gust.Models.GAT_Transformer import return_adj_matrix
def seed_everything(seed=42):
    """
    Seed everything to make the code more reproducible.
    :param seed: The seed for the random number generators.
    """
    random.seed(seed)         # Python random module.
    np.random.seed(seed)      # Numpy module.
    torch.manual_seed(seed)   # PyTorch.
    torch.cuda.manual_seed(seed)  # PyTorch CUDA; for GPU computation.
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # uses only deterministic convolution algorithms.
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be overridden.
# 使用此函数来固定所有需要的种子
seed_everything(42)
def train_decorator(func):
    def wrapper(self, *args, **kwargs):
        epochs = self.yaml_dict["Train"]["epoch"]
        device = self.yaml_dict["Train"]["device"]
        dataloader = self.train_loader
        self.model.train()  # Set model to training mode
        self.model = self.model.cuda()# 将模型移动到选择的设备上
        patience = self.yaml_dict["Train"]["patience"]
        ##train loss
        train_loss = []

        #optimizer
        optimizer = self.optimizer
        criterion = self.criterion
       
        best_loss = float('inf')
        for epoch in tqdm(range(epochs), desc='Training Progress'):
            running_loss = 0.0
            val_loss = 0.0
            for batch_idx, batch_data in enumerate(dataloader):
                #data(batch_size, seq_len,sensors)
                #adj(sensors,sensors)
                ## add judge in the exprimenal, the data (data,adj,aoa,target)
                if len(batch_data) == 4:
                    data, adj, aoa, target = batch_data
                    aoa = aoa.permute(1,0,2)#[4,2500,1]to[2500,4,1]
                    aoa = aoa.to(device)
                    # Do something with the 4 returned values
                elif len(batch_data) == 3:
                    data, adj, target = batch_data   
                    aoa = None   
      
              
                data, adj = data.to(device), adj[0,:,:].to(device)
                target = target.to(device)
                optimizer.zero_grad()
             
                output = self.model(data, adj, aoa)
           
                loss = criterion(output, target)
             
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            average_loss = running_loss / len(dataloader)
            print(f'Train Epoch: {epoch} Loss: {average_loss:.4f}')
            train_loss.append(average_loss)
            self.model.eval()  # Set the model to evaluation mode
            gc.collect()
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(self.test_loader):
                    if len(batch_data) == 4:
                        data, adj, aoa, target = batch_data
                        aoa = aoa.permute(1,0,2)#[4,2500,1]to[2500,4,1]
                        aoa = aoa.to(device)
                        # Do something with the 4 returned values
                    elif len(batch_data) == 3:
                        data, adj, target = batch_data   
                        aoa = None   
                    data, adj = data.to(device), adj[0,:,:].to(device)
                    target = target.to(device)

                    output = self.model(data, adj,aoa)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            val_loss /= len(self.val_loader)
            print(f'Validation Loss: {val_loss:.4f}')

            # 早停逻辑
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                #保存
                self.save_model()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs!')
                    break

            self.model.train()  # Make sure to reset to train mode after validation
        # 保存为 .npy 文件
        np.save(f'{self.save_dir}/train_loss.npy', train_loss)

    return wrapper

class Trainer:
    def __init__(self,
                 yaml_path:str
                 ):
        '''
        save_dir: could e.g. simulation_forward or chen_reverse or chen_forward_noise_test
        '''
        self.criterion = calculate_rmse
        
        self.yaml_dict = self._read_yaml(yaml_path)
        self.device = self.yaml_dict["Train"]["device"]
        self.save_dir =  self.yaml_dict["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)
        # 设置日志配置，日志保存到指定文件，同时设置了日志级别和日志格式
        logging.basicConfig(filename=f'{self.save_dir}/app.log', filemode='a', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # 使用logger记录信息
        logging.info('This is an informational message')
        self.best_loss = float('inf')
        
        self.model = self.prepare_model()
        self.load_dataloader()
        # 训练初始化
        self.optimizer = optim.Adam(self.model.parameters(), lr= float(self.yaml_dict["Train"]["lr"]) )
        # T_max是半个余弦周期的迭代次数，之后学习率会重置
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=40, eta_min=1e-6)

        logging.info("init finished")
   
    def _read_yaml(self,file_path) -> Optional[Dict]:
      with open(file_path, 'r') as file:
         data = yaml.safe_load(file)  # 使用safe_load而不是load
         # 检查 data 是否为字典类型
      if isinstance(data, dict):
         return data
      else:
         return None  # 或者你可以选择抛出一个异常，或记录一个错误日志
      
    def prepare_model(self)->nn.Module:
      
      if "GAT_Transformer" == self.yaml_dict["Model"]["name"]:
          
        from Gust.Models.GAT_Transformer import TransformerDecoder
        # 创建GAT_Transformer类的实例
        parameters = self.yaml_dict["Model"]["details"]
        model = TransformerDecoder(**parameters)
        print("model",model)
      else:
        model = None
      return model

      
    @train_decorator
    def train(self):
        '''
            call @train_decorator
        '''
        print("train")
          
    def test(self):
        '''
             测试+画图
        '''
        self.load_model(path=self.yaml_dict["load"]["ckpt"])
        self.model.eval()
        dataloader = self.test_loader
        device = self.device
        test_loss = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                              
                if len(batch_data) == 4:
                    data, adj, aoa, target = batch_data
                    aoa = aoa.permute(1,0,2)#[4,2500,1]to[2500,4,1]
                    aoa = aoa.to(device)
                    # Do something with the 4 returned values
                elif len(batch_data) == 3:
                    data, adj, target = batch_data   
                    aoa = None   
                data, adj = data.to(device), adj[0,:,:].to(device)
                target = target.to(device)
                output = self.model(data, adj,aoa) #[batch,seq_len,3]
                loss = criterion(output, target)
                test_loss = loss.item() +test_loss
                
                
            test_loss = test_loss/len(dataloader)
            # 创建一个2x1的子图布局
            fig, axs = plt.subplots(2, 1)

            # 第1个子图
            axs[0].plot(output[0,:].detach().cpu().numpy(),linestyle='--')
            axs[0].plot(target[0,:].detach().cpu().numpy())
            axs[0].set_title('Test')

            # 第2个子图,注意batch=1
            
            axs[1].imshow(self.model._get_layer_attention()[0,:,:].detach().cpu().numpy())
            axs[1].set_title('Attention')
            # 隐藏x轴和y轴的刻度标签
            # 隐藏x轴和y轴的刻度标签
            axs[1].xaxis.set_visible(False)
            axs[1].yaxis.set_visible(False)
            
            # 调整每个子图之间的间隔
            fig.tight_layout()
            plt.savefig("test_simu.png",dpi=300)
            print(f"test_loss:{test_loss}")
            logging.info(f"test_loss:{test_loss}")


    def save_model(self):
       
        filename = os.path.join(self.save_dir, 'best_model.pth')
        torch.save(self.model.state_dict(), filename)
        logging.info(f'New best model saved at {filename} ')
        print("save model")


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        logging.info(f'Model loaded from {path}')
        print("load_model!")
    
    def load_dataloader(self):
        '''
            载入数据,根据任务
        '''
        if self.yaml_dict["Task"] in ("Simulation_forward_train"):
            # 载入dataset   
            data_path = self.yaml_dict["Dataset_path"]
            dict_data =  torch.load(data_path)
            train,val,test = dict_data["train_dataset"],dict_data["valid_dataset"],dict_data["test_dataset"]
            # 创建DataLoader来批处理数据
            batch_size = self.yaml_dict["Train"]["batch"]
            self.train_loader = DataLoader(train, batch_size = batch_size, shuffle=True)
            self.val_loader = DataLoader(val, batch_size = batch_size, shuffle=True)
            self.test_loader = DataLoader(test, batch_size = batch_size, shuffle=True)
            print("loader,should 3, input,corrd,output",len(next(iter(self.train_loader))))
        
        elif self.yaml_dict["Task"] in ("Expr_forward_train","Expr_inverse_train"):
            # 载入dataset   
            data_path = self.yaml_dict["Dataset_path"]
            dict_data =  torch.load(data_path)
            
            train,val,test = dict_data["train_dataset"],dict_data["valid_dataset"],dict_data["test_dataset"]
            # 创建DataLoader来批处理数据
            batch_size = self.yaml_dict["Train"]["batch"]
            self.train_loader = DataLoader(train, batch_size = batch_size, shuffle=True)
            self.val_loader = DataLoader(val, batch_size = batch_size, shuffle=True)
            self.test_loader = DataLoader(test, batch_size = 1, shuffle=True)
            print("loader,should 3 or 4, input,corrd,output",len(next(iter(self.train_loader))))

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a YAML configuration from a command line string.")
    parser.add_argument('--yaml_path', type=str, help='Input configuration as YAML string')
    args = parser.parse_args()
    trainer = Trainer(yaml_path=args.yaml_path)
    # 源文件路径
    source_file = args.yaml_path
    destination_folder = trainer.yaml_dict["save_dir"]+"/config.yaml"
    # 使用 shutil.copy() 复制文件
    shutil.copy(source_file, destination_folder)
    if "train" in trainer.yaml_dict["Task"]:
        trainer.train()
        trainer.test()
    elif "test" in trainer.yaml_dict["Task"]:
        trainer.test()