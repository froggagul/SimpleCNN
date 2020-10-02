from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

# Training settings
# 만약 GPU를 사용 가능하다면 device 값이 cuda가 되고, 아니라면 cpu가 됩니다.
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')

# MNIST Dataset, 자동으로 다운로드 됨
# train용 데이터셋
train_dataset = datasets.MNIST(root='./mnist_data/', # download할 폴더 위치
                               train=True, # True를 지정하면 훈련 데이터로 다운로드
                               transform=transforms.ToTensor(),
                               download=True)
# test용 데이터셋
test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
batch_size = 64
train_loader = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True) # 왜 shuffle을 해주는 이유는 과적합 방지를 위해서

test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 첫번째층
        # image shape: (28, 28, 1)
        # conv layer: (24, 24, 10)
        # pool layer: (12, 12, 10)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        # 두번째층
        # image shape: (12, 12, 10)
        # conv layer: (8, 8, 20)
        # pool layer: (4, 4, 20)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layer층
        # linear 형태로 변환되므로, 4 * 4 * 20이 fc의 input size이다.
        self.fc = nn.Linear(320, 10) # 320->10의 full connected layer
        # 앞에 올 수가 확실치 않으면 임의의 값을 넣고 error를 관찰 

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # tensor를 linear 형태로 변환
        x = self.fc(x)
        return F.log_softmax(x, -1) # F.log_softmax(x) is deprecated, dimension을 명시해주어야 한다.


model = Net()
model.to(device) # 어차피 안되지만.. 나중을 위해 구현이라도..:)
criterion = nn.CrossEntropyLoss() # Loss function (~=cost function)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # optimzer


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # 기울기 초기화
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # back propagation, 역전파
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')


if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')