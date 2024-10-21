import sys
import numpy as np
import torch
import torch.nn.functional as F


class Logger(object):
    def __init__(self, fileN="default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train(epoch, network, train_loader, optimizer, scheduler, log, log_interval, path_name):
    network.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        _, prediction = torch.max(output.data, 1)
        correct += (prediction == target).sum().item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        assert not torch.isnan(loss), "Loss is NaN"

    print("Training accuracy: ", correct / len(train_loader.dataset) * 100)
    if log:
        torch.save(network.state_dict(), path_name + 'model.pth')
        torch.save(optimizer.state_dict(), path_name + 'optimizer.pth')
        torch.save(scheduler.state_dict(), path_name + 'scheduler.pth')


def test(test_loader, network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.type(torch.FloatTensor).cuda(), target.cuda()
            
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\n'+'Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)) + '\n')