import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from data.util import bgr2ycbcr

def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
def rgb_to_ycbcr(img):
    # 确保输入是一个四维张量 (B, C, H, W)
    assert img.dim() == 4 and img.size(1) == 3, "Input should be a 4D tensor with 3 channels"
    
    # 获取每个通道
    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    # 计算 Y, Cb, Cr 通道
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

    # 堆叠回一个张量 (B, C, H, W)
    ycbcr = torch.stack((y, cb, cr), dim=1)

    return ycbcr

def ycbcr_to_rgb(img):
    
    assert img.dim() == 4 and img.size(1) == 3, "Input should be a 4D tensor with 3 channels"
    
    
    y = img[:, 0, :, :]
    cb = img[:, 1, :, :] - 0.5
    cr = img[:, 2, :, :] - 0.5

    
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb

    
    rgb = torch.stack((r, g, b), dim=1)

    
    rgb = torch.clamp(rgb, 0.0, 1.0)

    return rgb


class threeCInvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(threeCInvBlock, self).__init__()
        self.channel_num=channel_num
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num -1

        self.clamp = clamp

        self.E = subnet_constructor(1, self.split_len1)
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)
        self.I = subnet_constructor(1, self.split_len2)
        self.J = subnet_constructor(1, self.split_len2)
        self.K = subnet_constructor(self.split_len1, 1)
        self.L = subnet_constructor(self.split_len2, 1)

    def forward(self, x, rev=False):
        x1, x2, x3 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2),x.narrow(1, self.channel_num-1, 1))

        if not rev:
            y1 = x1 + self.E(x3)+self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y3 = x3 + self.K(y1)+self.L(x2)
            self.t = self.clamp * (torch.sigmoid(self.J(y3)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s+self.t)) + self.G(y1)+self.I(y3)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            self.t = self.clamp * (torch.sigmoid(self.J(x3)) * 2 - 1)
            y2 = (x2 - self.G(x1)-self.I(x3)).div(torch.exp(self.s+self.t))
            y3 = x3 - self.K(x1)-self.L(y2)
            y1 = x1 - self.F(y2)-self.E(y3)

        return torch.cat((y1, y2, y3), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1
        
        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac



class TIRN_2(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, e_blocks=3, down_num=2):
        super(TIRN_2, self).__init__()
        current_channel = channel_in
        operations = []
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)

        operations = []
        for j in range(e_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.down_operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        jacobian = 0
        if not rev:
            out=x
            
            for op in self.haar_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in self.down_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            rgb=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))   
        else:
            out = x
            
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            
            for op in reversed(self.down_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            rgb=out[:,:3,:,:]
        if cal_jacobian:
            return out, rgb, jacobian
        else:
            return out, rgb

class TIRN_4(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, e_blocks=3, v_blocks=4, down_num=2):
        super(TIRN_4, self).__init__()
        current_channel = channel_in
        operations = []
        for i in range(1):
            b = HaarDownsampling(current_channel)
            
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)

        operations = []
        for j in range(e_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.down_operations = nn.ModuleList(operations)

        operations = []
        for i in range(1):
            b = HaarDownsampling(current_channel)
            
            operations.append(b)
            current_channel *= 4
        self.haar_operations_2 = nn.ModuleList(operations)

        operations = []
        for j in range(v_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.down_operations_2 = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        jacobian = 0
        if not rev:
            out=x      
            for op in self.haar_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev) 
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in self.down_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in self.haar_operations_2:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev) 
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in self.down_operations_2:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            rgb=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
        else:
            out = x
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in reversed(self.down_operations_2):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in reversed(self.haar_operations_2):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in reversed(self.down_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            #rgb=out[:,:3,:,:]
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            rgb=out[:,:3,:,:]
        if cal_jacobian:
            return out, rgb, jacobian
        else:
            return out, rgb



class TSAIN_2(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, e_blocks=3, v_blocks=1, down_num=2):
        super(TSAIN_2, self).__init__()
        current_channel = channel_in

        operations = []
        for i in range(down_num):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)

        operations = []
        for j in range(e_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.down_operations = nn.ModuleList(operations)
        
        operations = []
        for k in range(v_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.comp_operations = nn.ModuleList(operations)


    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.haar_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in self.down_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            LR=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
            for op in self.comp_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            LR_hat=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
            out=torch.cat((LR_hat,out[:,3:,:,:]),dim=1)
        else:
            out=x
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in reversed(self.comp_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            LR=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
            for op in reversed(self.down_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, LR, jacobian
        else:
            return out, LR

class TSAIN_4(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, e_blocks=3, v_blocks=3, down_num=2, f_blocks=2):
        super(TSAIN_4, self).__init__()
        current_channel = channel_in
        operations = []
        for i in range(1):
            b = HaarDownsampling(current_channel)
            
            operations.append(b)
            current_channel *= 4
        self.haar_operations = nn.ModuleList(operations)

        operations = []
        for j in range(e_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.down_operations = nn.ModuleList(operations)

        operations = []
        for i in range(1):
            b = HaarDownsampling(current_channel)
            
            operations.append(b)
            current_channel *= 4
        self.haar_operations_2 = nn.ModuleList(operations)

        operations = []
        for j in range(v_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.down_operations_2 = nn.ModuleList(operations)

        operations = []
        for k in range(f_blocks):
            b = threeCInvBlock(subnet_constructor, current_channel, channel_out-1)
            operations.append(b)
        self.comp_operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        jacobian = 0
        if not rev:
            out=x      
            for op in self.haar_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev) 
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in self.down_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in self.haar_operations_2:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev) 
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in self.down_operations_2:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            LR=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
            for op in self.comp_operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            LR_hat=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
            out=torch.cat((LR_hat,out[:,3:,:,:]),dim=1)
        else:
            out = x
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in reversed(self.comp_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            LR=ycbcr_to_rgb(torch.cat((out[:,-1:,:,:],out[:,:2,:,:]),dim=1))
            for op in reversed(self.down_operations_2):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in reversed(self.haar_operations_2):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            yuv=rgb_to_ycbcr(out[:,:3,:,:])
            uv=yuv[:,1:,:,:]
            y=yuv[:,:1,:,:]
            out=torch.cat((uv,out[:,3:,:,:],y),dim=1)
            for op in reversed(self.down_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            out=torch.cat((out[:,-1:,:,:],out[:,:-1,:,:]),dim=1)
            rgb=ycbcr_to_rgb(out[:,:3,:,:])
            #rgb=out[:,:3,:,:]
            out=torch.cat((rgb,out[:,3:,:,:]),dim=1)
            for op in reversed(self.haar_operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
            
        if cal_jacobian:
            return out, LR, jacobian
        else:
            return out, LR