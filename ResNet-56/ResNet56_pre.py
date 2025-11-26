# train_resnet56.py
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet56 import ResNet56  # 复用上面的模型

model = ResNet56(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

for epoch in range(160):
    for x, y in trainloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 40 == 0:
        torch.save(model.state_dict(), f'resnet56_cifar10_epoch{epoch+1}.pth')