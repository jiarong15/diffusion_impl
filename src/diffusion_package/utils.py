
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset

# def get_data(args):
#     transforms = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(80), 
#         torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#     return dataloader


def get_data_loader(batch_size, is_data_loader_shuffle):
    ds = load_dataset("uoft-cs/cifar10")

    transform = transforms.Compose([transforms.ToTensor()])
    tensor_x = torch.stack(list(map(lambda PILimg: transform(PILimg), ds['train']['img'])))
    tensor_y = torch.Tensor(ds['train']['label'])

    my_dataset = TensorDataset(tensor_x, tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset,
                               batch_size=batch_size,
                               shuffle=is_data_loader_shuffle)
    return my_dataloader

