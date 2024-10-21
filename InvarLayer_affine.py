import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf.basis import get_rbf


class InvarLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(InvarLayer, self).__init__()
        self.conv = torch.nn.Conv2d(6 * in_channels - 3, out_channels, kernel_size=1)
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

        uxx_uy = uxx * uy
        uxy_ux = uxy * ux
        uxy_uy = uxy * uy
        uyy_ux = uyy * ux
        
        inv2 = uxx * uyy - uxy * uxy

        inv3 = uxx_uy * uy - 2 * uxy_ux * uy + uyy_ux * ux
        inv4 = uxx_uy[:,:-1,:,:] * uy[:,1:,:,:] - uxy_ux[:,:-1,:,:] * uy[:,1:,:,:] - uxy_uy[:,:-1,:,:] * ux[:,1:,:,:] + uyy_ux[:,:-1,:,:] * ux[:,1:,:,:]
        inv5 = uxx_uy[:,1:,:,:] * uy[:,:-1,:,:] - uxy_ux[:,1:,:,:] * uy[:,:-1,:,:] - uxy_uy[:,1:,:,:] * ux[:,:-1,:,:] + uyy_ux[:,1:,:,:] * ux[:,:-1,:,:]
        inv345 = torch.cat((inv3, inv4, inv5), 1)
        inv6 = ux[:,:-1,:,:] * uy[:,1:,:,:] - uy[:,:-1,:,:] * ux[:,1:,:,:]

        inv1 = self.normalize(u, batch)
        inv2 = self.normalize(inv2, batch)
        inv345 = self.normalize(inv345, batch)
        if u.shape[1] > 1:
            is_batch_nonzero = (inv6 != 0).any(dim=1).any(dim=1).any(dim=1)
            max_values, _ = inv6.abs().view(batch, -1).max(dim=-1)
            max_values[~is_batch_nonzero] = 1.0
            inv6 = inv6.permute(3,1,2,0) / max_values
            inv6 = inv6.permute(3,1,2,0)          

        return torch.cat((inv1, inv2, inv345, inv6), 1)
    
    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.compute_inv(x, batch)
        x = self.conv(x)
        x = F.relu(x)
        if self.stride == 2:
            x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = InvarLayer(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = InvarLayer(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class InvarLayer_affine_resnet32(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[5, 5, 5], num_classes=10):
        super(InvarLayer_affine_resnet32, self).__init__()
        self.in_planes = 16
        self.conv1 = InvarLayer(1, 16, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return torch.log_softmax(out, dim=1)