
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset


def get_data_loader(batch_size, is_data_loader_shuffle):

    ## Load the cifar dataset for a set of images 
    ## from huggingface for testing purposes
    ds = load_dataset("uoft-cs/cifar10")

    transform = transforms.Compose([transforms.ToTensor()])
    tensor_x = torch.stack(list(map(lambda PILimg: transform(PILimg), ds['train']['img'])))
    tensor_y = torch.LongTensor(ds['train']['label'])

    my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset,
                               batch_size=batch_size,
                               shuffle=is_data_loader_shuffle)
    return my_dataloader

