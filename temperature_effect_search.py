from checking_sampling import check_with_model
import torch


if __name__ == '__main__':
    
    torch.manual_seed(42)
    model_name = "size_256model_TrainedOn0_final1.dat"

    for t in range(-20, 2, 1):
        print(2**(t/10))
        #check_with_model(model_name, sample_size = 128000, Te = 2**(t/10))
        
    
    
    