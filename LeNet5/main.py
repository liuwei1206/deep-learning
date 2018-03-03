#author = liuwei
import gzip, struct
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import math

#读取数据的函数,先读取标签，再读取图片
def _read(image, label):
    minist_dir = 'data/'
    with gzip.open(minist_dir + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(minist_dir + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return image, label 

#读取数据
def get_data():
    train_img, train_label = _read(
	    'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')

    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [train_img, train_label, test_img, test_label]


#定义lenet5
class LeNet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        #定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        #第二个卷积层，6个输入，16个输出，5*5的卷积filter 
        self.conv2 = nn.Conv2d(6, 16, 5)

        #最后是三个全连接层
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''前向传播函数'''
        #先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        #重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.view(-1, self.num_flat_features(x))
        #print('size', x.size())
        #第一个全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        #x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
        size = x.size()[1:]        #x.size返回的是一个元组，size表示截取元组中第二个开始的数字
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

#定义一些超参数
use_gpu = torch.cuda.is_available()
batch_size = 256
kwargs = {'num_workers': 2, 'pin_memory': True}                              #DataLoader的参数

#参数值初始化
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()

#训练函数
def train(epoch):
    #调用前向传播
    model.train()		
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)                      #定义为Variable类型，能够调用autograd
        #初始化时，要清空梯度
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()                                                     #相当于更新权重值
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.data[0]))

#定义测试函数			   
def test():
    model.eval()                                                             #让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        #计算总的损失
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]                           #获得得分最高的类别
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

#获取数据，
X, y, Xt, yt = get_data()

train_x, train_y = torch.from_numpy(X.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(y.astype(int))
test_x, test_y = [torch.from_numpy(Xt.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(yt.astype(int))]

#封装好数据和标签
train_dataset = TensorDataset(data_tensor=train_x, target_tensor=train_y)
test_dataset = TensorDataset(data_tensor=test_x, target_tensor=test_y)

#定义数据加载器
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, **kwargs)
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, **kwargs)

#实例化网络
model = LeNet5()
if use_gpu:
    model = model.cuda()
    print('USE GPU')
else:
    print('USE CPU')

#定义代价函数，使用交叉熵验证
criterion = nn.CrossEntropyLoss(size_average=False)
#直接定义优化器，而不是调用backward
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

#调用参数初始化方法初始化网络中的所有参数
model.apply(weight_init)                                                      #了解apply用法

#调用函数执行训练和测试
for epoch in range(1, 501):
    print('----------------start train-----------------')
    train(epoch)
    print('----------------end train-----------------')

    print('----------------start test-----------------')
    test()
    print('----------------end test-----------------')
