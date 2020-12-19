from checking_sampling import check_with_model
import torch


if __name__ == '__main__':
    
    torch.manual_seed(42)
    model_name = "size_512model_TrainedOn0_final1.dat"
    '''
    check_with_model(model_name, sample_size = 256 * 64)
    check_with_model(model_name, sample_size = 256 * 128)
    check_with_model(model_name, sample_size = 256 * 256)
    check_with_model(model_name, sample_size = 256 * 512)
    check_with_model(model_name, sample_size = 256 * 1024)
    check_with_model(model_name, sample_size = 256 * 2048)
    '''

    check_with_model(model_name, sample_size = 10000, Te = 0.43)
    check_with_model(model_name, sample_size = 100000, Te = 0.43)
    check_with_model(model_name, sample_size = 1000000, Te = 0.43)