import torch
import torch.nn as nn
import time
import numpy as np

__all__ = ['C2REPA']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    # default_act = nn.SiLU()  # default activation
    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


def conv_bn(c1, c2, k=1, s=1, p=None, g=1, d=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False))
    result.add_module('bn', nn.BatchNorm2d(c2))

    return result

class RepVGGBlock(nn.Module):
    default_act = nn.ReLU()

    def __init__(self, dim, deploy=False, act=True):
        super().__init__()

        self.c = dim
        self.c_half = dim // 2
        self.deploy = deploy
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(self.c, self.c, 3, 1, 1, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(self.c)
            self.rbr_3x3 = conv_bn(self.c, self.c, 3)
            self.rbr_1x1 = conv_bn(self.c, self.c, 1)

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_1x1'):
            kernel, bias = self.get_equivalent_kernel_bias()

            self.rbr_reparam = nn.Conv2d(self.c, self.c, 3, 1, 1, bias=True)
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias

            for para in self.parameters():
                para.detach_()

            # self.rbr_3x3 = self.rbr_reparam
            self.__delattr__('rbr_3x3')
            self.__delattr__('rbr_1x1')
            if hasattr(self, 'rbr_identity'):
                self.__delattr__('rbr_identity')
            if hasattr(self, 'id_tensor'):
                self.__delattr__('id_tensor')
            self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                kernel_value = np.zeros((self.c, self.c, 3, 3), dtype=np.float32)
                for i in range(self.c):
                    kernel_value[i, i, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - (running_mean * gamma / std)

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        else:
            return self.act(self.rbr_identity(x) + self.rbr_3x3(x) + self.rbr_1x1(x))


class C2REPA(nn.Module):
    default_act = nn.ReLU()
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1)
        self.m = nn.ModuleList(RepVGGBlock(self.c) for _ in range(n))

    def switch_to_deploy(self):
        for module in self.m:
            module.switch_to_deploy()

    def forward(self, x):
        x = self.cv1(x)

        for layer in self.m:
            x = layer(x)

        return self.cv2(x)


if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)

    # Model
    model = C2REPA(64, 128, n=4)

    model.eval()

    t1 = time.time()
    out1 = model(image)
    t2 = time.time()
    print(out1.shape)

    model.switch_to_deploy()

    t3 = time.time()
    out2 = model(image)
    t4 = time.time()
    print(out2.shape)

    print(torch.allclose(out1, out2, atol=1e-5))

    print(t2-t1)
    print(t3-t2)
