#author = liuwei
import alexnet

import torch
import torch.nn as nn
import torch.optim

use_gpu = torch.cuda.is_available()

def main(model_name):
    if model_name == 'alexnet':
        model = AlexNet()
        input_size = 227

    if use_gpu:
        model = model.cuda()
        print('USE GPU')
    else:
        print('USE CPU')

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters, lr=0.001, momnetum=0.1, weight_decay=0.1)

    #训练和测试的数据集呢？
    
    epoch = 20

def train(train_loader, model, criterion, optimizer, epoch):
    '''params include filepath, which model, '''
    #switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        if use_gpu:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)                #set to can be train

        #get output and cal loss
        output = model(input)
        loss = criterion(output, target)

        #update all the param
        optimizer.zero_grad()                                            #init the grad
        loss.backward()                                                  #backward process
        optimizer.step()                                                 #update weight

        #print process
        if i % 100 == 0:
            print('Train　Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  i, i * len(input), len(train_loader.dataset), 
                  100. * i / (len(train_loader)), loss.data[0]))

def validate(test_loader, model, criterion):
    model.eval()

    for i, (input, target) in enumerate(test_loader):
        if use_gpu:
            input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target)

        output = model(input)                                            #the predict res
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1] 
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()         #本轮正确的个数
    
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
    	  test_loss, correct, len(test_loader), 100. * correct / len(test_loader)))


    

