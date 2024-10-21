import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf.basis import get_rbf


class InvarLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(InvarLayer, self).__init__()
        self.conv = torch.nn.Conv2d(6 * in_channels, out_channels, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1)
        kernel1 = self.make_gauss(1, 7).cuda().reshape(2, 1, 1, 7, 7)
        kernel2 = self.make_gauss(2, 7).cuda().reshape(3, 1, 1, 7, 7)
        self.conv_weights = [kernel1[0], kernel1[1], kernel2[0], kernel2[2]]
        self.stride = stride

    def dx(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[0]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=3, stride=1, groups=c)

    def dy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[1]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=3, stride=1, groups=c)
    
    def dxx(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[2]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=3, stride=1, groups=c)

    def dyy(self, u):
        _,c,_,_ = u.shape
        weights = torch.cat([self.conv_weights[3]]*c, 0)
        return F.conv2d(u, weights, bias=None, padding=3, stride=1, groups=c)

    def make_coord(self, kernel_size):
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        coord = torch.meshgrid([-x, x])
        coord = torch.stack([coord[1], coord[0]], -1)
        return coord.reshape(kernel_size ** 2, 2)

    def make_gauss(self, order, kernel_size):
        diff = []
        coord = self.make_coord(kernel_size)
        gauss = get_rbf('ga')
        for i in range(order + 1):
            w = gauss(coord, torch.zeros(1, 2), eps=0.99, diff=[i, order - i]).reshape(kernel_size, kernel_size)
            w = torch.tensor(w)
            diff.append(w)
        tensor = torch.stack(diff, 0)
        return tensor.to(torch.float32)

    def normalize(self, inv, batch):
        inv = inv.permute(3,1,2,0) / inv.abs().view(batch, -1).max(dim=-1)[0]
        inv = inv.permute(3,1,2,0)
        return inv
    
    def compute_inv(self, u, batch):
        ux = self.dx(u)
        uy = self.dy(u)
        uxx = self.dxx(u)
        uyy = self.dyy(u)
        uxy = self.dx(uy)

        inv0 = self.normalize(u ,batch)
        inv1 = self.normalize(torch.cat((ux, uy), 1), batch)
        inv2 = self.normalize(torch.cat((uxx, uxy, uyy), 1), batch)
        return torch.cat((inv0, inv1, inv2), 1)
    
    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.compute_inv(x, batch)
        x = self.conv(x)
        x = F.relu(x)
        if self.stride == 2:
            x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        return x


class InvarLayer_scale_cnn(nn.Module):
    def __init__(self, C1=83, C2=163, C3=247):
        super().__init__()

        self.main = nn.Sequential(
            InvarLayer(1, C1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),
            
            InvarLayer(C1, C2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),
            
            InvarLayer(C2, C3),
            nn.ReLU(True),
            nn.MaxPool2d(7),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return torch.log_softmax(x, dim=1)