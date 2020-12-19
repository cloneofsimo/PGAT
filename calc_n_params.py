import torch

model_128 = "size_128model_TrainedOn0_final1.dat"
model_256 = "size_256model_TrainedOn0_final1.dat"
model_512 = "size_512model_TrainedOn0_final1.dat"

def get_n_params(mn):
    model = torch.load(mn)
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    
    print(mn, pp)

get_n_params(model_128)

get_n_params(model_256)

get_n_params(model_512)