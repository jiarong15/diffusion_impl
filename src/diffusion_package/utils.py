import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from datasets import load_dataset


def get_data_loader(batch_size, is_data_loader_shuffle):

    def get_canny_edges(PILimg):
        trans_img = np.asarray(PILimg)
        (B, G, R) = cv2.split(trans_img)
        B_cny = cv2.Canny(B, 100, 200)
        G_cny = cv2.Canny(G, 100, 200)
        R_cny = cv2.Canny(R, 100, 200)
        edges = cv2.merge([B_cny, G_cny, R_cny])
        edges = np.transpose(edges, (2, 0, 1))
        return edges

    ## Load the cifar dataset for a set of images 
    ## from huggingface for testing purposes
    ds = load_dataset("uoft-cs/cifar10")

    ## Extra operations to convert data into a dataset and
    ## fed in dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = torch.stack(list(map(lambda PILimg: transform(PILimg), ds['train']['img'])))
    label_tensor = torch.LongTensor(ds['train']['label'])
    edge_tensor = torch.stack(list(map(lambda PILimg: torch.tensor(get_canny_edges(PILimg)).float(), ds['train']['img'])))

    my_dataset = TensorDataset(img_tensor, label_tensor, edge_tensor) # create your datset
    my_dataloader = DataLoader(my_dataset,
                               batch_size=batch_size,
                               shuffle=is_data_loader_shuffle)
    return my_dataloader

