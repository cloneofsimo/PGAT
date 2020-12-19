import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm

from Password_dataset import Password_dataset
from TransformerPGA import TransformerPGA

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


PA_set = [1, 2, 3, 4]

Metric_pt = [[0 for _ in range(4)] for _ in range(4)]
Metric_jt = [[0 for _ in range(4)] for _ in range(4)]


def train(model, PAset, N = 2000000):
    device = torch.device("cuda:0")
    epochs = 1
    batch_size = 256
    lr = 2e-5
    max_len = 10
    chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
    n_vocab = len(chars) + 2
    
    opt = optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-10)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, step_size_up= 1000, base_lr= lr, max_lr= lr * 5, cycle_momentum= False)
    dataset = Password_dataset(max_len = max_len, chars = chars, types = str(PAset), set_limit= N)
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
            
        print(model.sample(n = 2))
        print(f'Epoch {epoch} : Loss : {tot_loss/cnt :.5f}')

def test(model, PAset, sample_size = 100000):
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
    sampled = []
    while len(sampled) < sample_size:
        sampled.extend(model.sample(n = 200, T = 1))
        #print(len(sampled))
    
    sampled = set(sampled)

    with open(f"PA_{PAset}.txt", 'r') as F_tr:
        whole = F_tr.read()
        whole = whole.split('\n')

        for word in whole:
            if len(word) > 10:
                print(word)
    cnt = 0
    for pw in whole:
        if pw in sampled:
            cnt += 1
    
    return cnt




if __name__ == "__main__":
    torch.manual_seed(0) 
#test how pretraining works
#pre-trained model should have 4 head, 512 dim, 1024 ffdim, 6 layers
    
    for Pa in PA_set:
        model = torch.load("pt_size_512model_TrainedOnEnglish_words_final1.dat")
        #print(model.sample(20))
        train(model, Pa)
        torch.save(model, f"ftOn{Pa}.dat")
        for Pb in PA_set:
            if Pa == Pb:
                continue
            
            res = test(model, Pb)
            
            Metric_pt[Pa-1][Pb-1] = res
            print(f"metric {Pa}{Pb} is {res}")


    torch.save(Metric_pt, "metric_pt.res")
    
    # test how non-pretraining works

    for Pa in PA_set:
        
        d_m = 512
        chars = list(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~")
        n_vocab = len(chars) + 2
        device = torch.device("cuda:0")
        max_len = 10
        model = TransformerPGA(
            d_model = d_m,
            n_head = 4,
            dim_feedforward= 2*d_m,
            num_layers = 6,
            n_vocab = n_vocab, 
            max_len = max_len, 
            chars = chars,
            device = device
        )
        train(model, Pa)
        torch.save(model, f"TrainedOn{Pa}.dat")
        
        for Pb in PA_set:
            if Pa == Pb:
                continue
            res = test(model, Pb)
            Metric_jt[Pa-1][Pb-1] = res
            print(f"metric {Pa}{Pb} is {res}")
    
    torch.save(Metric_jt, "metric_jt.res")

'''
Res:

MAX IN LNS Y :  10
current loss : 2.64800: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [22:36<00:00,  5.76it/s]
['cincent81', 'MRanger#1']
Epoch 1 : Loss : 2.64800
metric 12 is 80276
metric 13 is 148294
metric 14 is 880
MAX IN LNS Y :  10
current loss : 2.11854: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [23:08<00:00,  5.63it/s]
['dancers', 'dessel123']
Epoch 1 : Loss : 2.11854
metric 21 is 6607
metric 23 is 730250
metric 24 is 2757
MAX IN LNS Y :  10
current loss : 1.92816: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [22:45<00:00,  5.72it/s]
['mommyj', 'bendinal']
Epoch 1 : Loss : 1.92816
metric 31 is 5679
metric 32 is 354427
metric 34 is 3012
MAX IN LNS Y :  10
current loss : 2.54279: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [23:07<00:00,  5.63it/s]
['kpihno', 'frodin']
Epoch 1 : Loss : 2.54279
metric 41 is 856
metric 42 is 15643
metric 43 is 21349
MAX IN LNS Y :  10
current loss : 2.71730: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [22:29<00:00,  5.79it/s]
['112003303', 'Hermado1#']
Epoch 1 : Loss : 2.71730
metric 12 is 44990
metric 13 is 143538
metric 14 is 639
MAX IN LNS Y :  10
current loss : 2.20107: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [22:59<00:00,  5.66it/s]
['picture1', 'ett0705']
Epoch 1 : Loss : 2.20107
metric 21 is 5947
metric 23 is 700124
metric 24 is 2621
MAX IN LNS Y :  10

metric 32 is 331060
metric 34 is 2687
MAX IN LNS Y :  10
current loss : 2.60120: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7812/7812 [23:53<00:00,  5.45it/s] 
['madisuvk68', '6712nale']
Epoch 1 : Loss : 2.60120
metric 41 is 714
metric 42 is 8192
Hypothesis : Pre trained 된 모델은 더 generalize 를 잘 하는가?

'''

