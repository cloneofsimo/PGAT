import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm

from Password_dataset import Password_dataset
from TransformerPGA import TransformerPGA

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    epochs = 2
    batch_size = 256
    lr = 1e-3
    max_len = 10
    types = "0"
    d_m = 32
    
    chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
    n_vocab = len(chars) + 2
    model = TransformerPGA(
        d_model = d_m,
        n_head = 4,
        dim_feedforward= 4*d_m,
        num_layers = 3,
        n_vocab = n_vocab, 
        max_len = max_len, 
        chars = chars,
        device = device
    )
    opt = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-10)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, step_size_up= 1000, base_lr= lr, max_lr= lr * 5, cycle_momentum= False)

    dataset = Password_dataset(max_len = max_len, chars = chars, types = types)
    dl = DataLoader(dataset, shuffle= True, batch_size= batch_size, drop_last= True, num_workers = 3)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl)
        tot_loss = 0
        cnt = 0
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
            scheduler.step()
            if cnt%2000 == 0:
                torch.save(model, f"size_{d_m}model_TrainedOn{types}_checkpoint_epoch_{epoch}mid{cnt//1000}.dat")
            
        print(model.sample(n = 500))
        print(f'Epoch {epoch} : Loss : {tot_loss/cnt :.5f}')

        torch.save(model, f"size_{d_m}model_TrainedOn{types}_final{epoch}.dat")