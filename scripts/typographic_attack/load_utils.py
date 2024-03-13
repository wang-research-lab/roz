import os
import clip
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet50, vgg19, resnet101, vgg16, alexnet, resnet152
from PIL import Image
import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
import pickle
import torch.nn as nn
from efficientnet_pytorch import EfficientNet




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('RN50', device)
import timm
def load_pretrained_vision_models(model_name):
    assert model_name in  \
        ['vgg19', 'resnet50', 'resnet101', 'vgg16', 'alexnet', 'resnet152', 'ViT-B']
    if model_name == 'ViT-B':
        model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)
        model.head = nn.Linear(768, 10)
    else:
        model = eval(model_name)(pretrained=False, )
        model.fc = nn.Linear(2048, 10)
    path = model_name + '.pkl'
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_pretrained_imagenet_models(model_name):
    assert model_name in \
           ['vgg19', 'resnet50', 'resnet101', 'vgg16', 'alexnet', 'vgg11', 'resnet18', 'resnet34',
            'RN50x4', 'ViT-B/32']
    if model_name == 'RN50x4':
        model = EfficientNet.from_pretrained('efficientnet-b3')
    elif model_name == 'ViT-B/32':
        import timm
        model = timm.create_model('vit_base_patch32_384', pretrained=True).cuda()
    else:
        model = eval(model_name)(pretrained=True, )
    model = model.to(device)
    return model.eval()






def get_clip(clip_type):
    return clip.load(clip_type, device, jit=False)[0].eval()


def get_transform():
    return preprocess




# def get_imagenet_synsets_info():


def get_CIFAR10_synsets_info():
    synsets_info_path = './CIFAR10/cifar-10-batches-py/batches.meta'
    import pickle
    with open(synsets_info_path, 'rb') as f:
        synsets_li = pickle.load(f)['label_names']
    return synsets_li

class MDataset(torch.utils.data.Dataset):
    def __init__(self, data_li):
        self.data_li = data_li

    def __len__(self):
        return len(self.data_li)

    def __getitem__(self, idx):
        return self.data_li[idx]


def get_CIFAR10_loader(batch_size=32, sample=True, sample_type=224):
    if sample:
        if sample_type == 224:
            with open('sample_cifar10.pickle', 'rb') as f:
                data_li = pickle.load(f)
        else:
            with open('sample_cifar10_RN50.pickle', 'rb') as f:
                data_li = pickle.load(f)
        return DataLoader(MDataset(data_li), shuffle=False, batch_size=batch_size)

    else:
        from torchvision.datasets import CIFAR10
        cifar10 = CIFAR10(root='./CIFAR10', train=False, transform=preprocess)
        return DataLoader(cifar10, shuffle=False, batch_size=batch_size)



def get_imageNet_loader(batch_size=32, sample=True, sample_type=224):
    if sample:
        if sample_type == 224:
            with open('sample_imgNet.pickle', 'rb') as f:
                data_li = pickle.load(f)
        elif sample_type == 288:
            with open('sample_imgNet_RN50.pickle', 'rb') as f:
                data_li = pickle.load(f)
        else:
            with open('sample_imgNet_vit.pickle', 'rb') as f:
                data_li = pickle.load(f)

        return DataLoader(MDataset(data_li), shuffle=False, batch_size=batch_size)
    else:
        imagenet_path = ImageNetPATH
        from torchvision.datasets import ImageNet
        imagenet = ImageNet(root=imagenet_path, split='val', transform=preprocess)
        return DataLoader(imagenet, shuffle=False, batch_size=batch_size)







def get_imageNet_synsets_info():
    synsets_info_path = './ILSVRC2012_devkit_t12/data/meta.mat'
    raw_data = sio.loadmat(synsets_info_path)['synsets']

    synsets_list = []

    for i, synset in enumerate(raw_data):
        if i == 1000:
            break
        orig_synset = synset[0]

        synset_id = orig_synset[1].item()
        synset_label = orig_synset[2].item()
        gloss = orig_synset[3].item()
        synsets_list.append((synset_id, synset_label, gloss))

    sorted_list = sorted(synsets_list, key=lambda item: item[0])
    synsets_dict = {}
    for i, (synset_id, synset_label, gloss) in enumerate(sorted_list):
        synsets_dict[synset_id] = (i, synset_label, gloss)

    return synsets_dict




