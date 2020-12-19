import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm

from Password_dataset import Password_dataset
from TransformerPGA import TransformerPGA

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda:0")

batch_size = 256
lr = 2e-4
max_len = 10
d_m = 32
chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
n_vocab = len(chars) + 2

dataset = Password_dataset(max_len = max_len, chars = chars, set_limit = 4000)
dl = DataLoader(dataset, shuffle = False, batch_size= batch_size, drop_last= True, num_workers = 0)


def LR_val(lr = 1e-1):
    criterion = nn.CrossEntropyLoss()
    
    model = TransformerPGA(
        d_model = d_m,
        dim_feedforward= 4*d_m,
        n_head = 4,
        num_layers = 3,
        n_vocab = n_vocab, 
        max_len = max_len, 
        chars = chars,
        device = device
    )
    opt = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-10)
    model.to(device)
    pbar = tqdm(dl)
    cnt = 0
    tot_loss = 0
    for (yin, yout) in pbar:
        yin = torch.cat([torch.ones(batch_size, 1) * (n_vocab - 1), yin], dim = 1).long()
        yin = yin.to(device)
        yout = yout.to(device)
        #print(yin, yout)
        y_pred = model(yin)

        loss = criterion(y_pred.view(-1, n_vocab - 1), yout.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_loss += loss.item()
        cnt += 1
        pbar.set_description(f"current loss : {tot_loss/cnt:.5f}")
        
    return tot_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    
    
    #chars = list("0987654321-+*()^xyz") (case with three variables)
    
    valset = {(10**(p/10),LR_val(lr = 10**(p/10))) for p in range(-5, -50, -1)}
    print(valset)