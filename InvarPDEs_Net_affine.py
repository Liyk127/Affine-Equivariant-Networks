import torch
import torch.nn as nn
import torch.nn.functional as F
from rbf.basis import get_rbf


def compute_num_inv(C):
    return 6 * C - 3


class InvarPDEs_Net_affine(torch.nn.Module):
    def __init__(self, num_class=10, channel=45, hidden_channel=45, iter=15):
        super(InvarPDEs_Net_affine, self).__init__()
        self.num_class = num_class
        self.channel = channel
        self.hidden_channel = hidden_channel
        self.iter = iter
        
        conv_list1 = []
        conv_list2 = []
        bn_list1 = []
        bn_list2 = []
        C = self.channel
        num_inv = compute_num_inv(C)
        for t in range(self.iter):
            if t == self.iter // 2:
                self.conv_lift1 = torch.nn.Conv2d(C, 2*C, kernel_size=1)
                C *= 2
                num_inv = compute_num_inv(C)
                self.bn_lift1 = nn.BatchNorm2d(C)
            conv_list1.append(torch.nn.Conv2d(num_inv, self.hidden_channel, kernel_size=1))
            conv_list2.append(torch.nn.Conv2d(self.hidden_channel, C, kernel_size=1))
            bn_list1.append(nn.BatchNorm2d(self.hidden_channel))
            bn_list2.append(nn.BatchNorm2d(C))
            
        self.conv_list1 = nn.ModuleList(conv_list1)
        self.conv_list2 = nn.ModuleList(conv_list2)
        self.bn_list1 = nn.ModuleList(bn_list1)
        self.bn_list2 = nn.ModuleList(bn_list2)
        self.fc1 = nn.Linear(C, 64)
        self.fc2 = nn.Linear(64, self.num_class)
        learnable_step = torch.tensor([2/self.iter]*self.iter, requires_grad=True)
        self.step = nn.Parameter(learnable_step)

        kernel1 = self.make_gauss(1, 7).cuda().reshape(2, 1, 1, 7, 7)
        kernel2 = self.make_gauss(2, 7).cuda().reshape(3, 1, 1, 7, 7)
        self.conv_weights = [kernel1[0], kernel1[1], kernel2[0], kernel2[2]]


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
        batch, data_channel, h, w = x.shape
        C = self.channel

        #initialization
        u = torch.empty((batch, C, h, w)).cuda()
        for i in range(C):
            u[:, i, :, :] = x[:, i % data_channel, :, :]
                
        for t in range(self.iter):
            if t == self.iter // 2:
                u = F.relu(self.bn_lift1(self.conv_lift1(u)))
                C *= 2

            inv = self.compute_inv(u, batch)
            inv = F.relu(self.bn_list1[t](self.conv_list1[t](inv)))
            inv = self.bn_list2[t](self.conv_list2[t](inv))
            u = F.relu(u + self.step[t] * inv)

        output = F.max_pool2d(u, (h,w))
        output = output.view(batch, C)
        output = self.fc2(F.relu(self.fc1(output)))

        return torch.log_softmax(output, dim=1)