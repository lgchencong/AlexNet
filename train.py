import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets
from model import AlexNet
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
print(device)
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
image_path = 'data/'
print(data_transform["train"])
train_datasets = datasets.ImageFolder(root=image_path + 'train', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]))
val_datasets = datasets.ImageFolder(root=image_path + 'val', transform=data_transform["val"])
train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=0)
net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
save_path = './AlexNet.pth'
for epoch in range(3):
    # train
    net.train()  # 训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
