from torchvision import datasets
from torchvision.models import vgg19, resnet50, resnet101, resnext50_32x4d, resnet152
import clip
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from vit_pytorch import ViT
from torch.nn.utils import clip_grad_norm_
import timm


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='resnet50')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.02, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--epoch_nums', default=50, type=int)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--save_model_name', default='')
parser.add_argument('--type', default='ViT')

params = parser.parse_args()

EPOCHS = params.epoch_nums
BATCH_SIZE = params.batch_size
device = "cuda" if torch.cuda.is_available() else "cpu"


vit_flag = True if params.type == 'ViT' else False


_, preprocess = clip.load("ViT-B/32", device=device)

train_set = datasets.CIFAR10(root='./CIFAR10', train=True, transform=preprocess, download=True)
test_set = datasets.CIFAR10(root='./CIFAR10', train=False, transform=preprocess, download=True)


head = None
if vit_flag:
    model = timm.create_model('vit_base_patch32_224_in21k', pretrained=True)
    model.head = nn.Linear(768, 10)
    head = model.head
else:
    model = eval(params.model_name)(pretrained=True)
    model.fc = nn.Linear(2048, 10, bias=True)
    head = model.fc

model = model.to(device)
def adjust_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    for param_group in optimizer.param_groups:
        adjusted_lr = lr / 1.5
        param_group['lr'] = adjusted_lr if adjusted_lr > 1e-5 else 1e-5



train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()
if params.optimizer == 'adam':
    optimizer = torch.optim.AdamW(head.parameters(), lr=params.lr, weight_decay=params.weight_decay)
else:
    optimizer = torch.optim.SGD(head.parameters(), lr=params.lr, weight_decay=params.weight_decay, momentum=0.9)







def evaluate():
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            if vit_flag:
                output = model(images, )
            else:
                output = model(images) # 32,10
            _, indices = torch.max(output, dim=1)  # 32
            correct = torch.sum(indices==labels).item()
            total_correct += correct
            total += labels.shape[0]
    return total_correct / total




def train():
    best_acc = -1
    global model
    best_model = None
    try:
        last_epoch_loss = 10000
        for epoch in range(EPOCHS):
            total_loss = 0
            model.train()
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)

                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
            # exp_lr_scheduler.step()
            acc = evaluate()
            if acc > best_acc:
                best_acc = acc
                best_model = model
            print('epoch: {}/{}, training_loss: {}/{}, acc: {}/{}'.format(epoch+1, EPOCHS,
                                                                          total_loss / len(train_loader),
                                                                          last_epoch_loss,
                                                                          acc, best_acc))
            print('*' * 89)
            if total_loss / len(train_loader) > last_epoch_loss:
                print('need to adjust lr, current lr: {}'.format(optimizer.param_groups[0]['lr']))
                adjust_lr(optimizer)
            last_epoch_loss = total_loss / len(train_loader)
    except KeyboardInterrupt:
        print('exists from training early')


    model = best_model
    save = input("Save ? (y/n)")
    if 'y' in save:
        torch.save(model.state_dict(), params.save_model_name)






if __name__ == '__main__':
    train()



