import torch
from torch.utils.data import Dataset

class Password_dataset(Dataset):
    def __init__(self, max_len = 10, chars = list("!\"#$%&'()*+,-./0123456789;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"), types = "0", set_limit = 99987654321):
        
        y = []
        PAD = max_len  
        
        self.stoi = {ch : i + 1 for i, ch in enumerate(chars)}
        self.itos = { i + 1 : ch for i, ch in enumerate(chars)}
        
        with open(f"PA_{types}.txt", 'r') as F_tr:
            whole = F_tr.read()
            whole = whole.split('\n')

            for word in whole:
                if len(word) > 10:
                    print(word)
            
            lns = max(list(map(len, whole)))
            print("MAX IN LNS Y : ", lns)
            for ele in whole:
                x_l = [self.stoi[v] for v in list(ele)]
                x_l.extend([0] * PAD)
                x_l = x_l[:PAD]
                y.append(x_l)
            
        self.y = torch.tensor(y, dtype = torch.long)
        self.len_ = min(set_limit, self.y.shape[0])

    
    def __len__(self):
        return self.len_
    
    def __getitem__(self, idx):
        return self.y[idx, :-1], self.y[idx,:]
        

                