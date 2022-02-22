import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets
import pdb
import tensorflow.keras.preprocessing as tf
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torchvision import transforms
test_dir = "IMAGES_224x224_SPLIT/val"
train_dir = "IMAGES_224x224_SPLIT/train"
IMG_SIZE = (224, 224)
batchsize = 16
num_epochs = 50

data_transforms = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456,0.406],
        [0.229, 0.224, 0.225])])

def getDataset():
    train_data = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batchsize,
            shuffle=True,
            num_workers=4)
            
    test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batchsize,
            shuffle=True,
            num_workers=4)
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = getDataset()
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() /inputs.size(0)
            running_corrects += torch.sum(preds == labels.data) / inputs.size(0)
            
        exp_lr_scheduler.step()
        train_epoch_loss = running_loss / len(train_loader)
        train_epoch_acc = running_corrects / len(train_loader)
            
        model.eval()
        running_loss = 0.0
        running_corrects = 0
            
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() / inputs.size(0)
            running_corrects += torch.sum(preds == labels.data) / inputs.size(0)
            
            
        epoch_loss = running_loss / len(test_loader)
        epoch_acc = running_corrects.double() / len(test_loader)
        print("Train: Loss: {:.4f} Acc: {:.4f}"
    " Val: Loss: {:.4f}"
    " Acc: {:.4f}".format(train_epoch_loss,
                          train_epoch_acc,
                          epoch_loss,
                          epoch_acc))
            
    torch.save(model.state_dict(), "./resnet34.pt")
    
