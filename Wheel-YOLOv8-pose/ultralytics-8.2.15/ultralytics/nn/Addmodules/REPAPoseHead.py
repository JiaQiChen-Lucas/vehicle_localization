import math
import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors
import numpy as np
import time

# classes

__all__ = ['REPADetectHead', 'REPAPoseHead']

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
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

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """
    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

def conv_bn(c1, c2, k=1, s=1, p=None, g=1, d=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False))
    result.add_module('bn', nn.BatchNorm2d(c2))

    return result

class RepVGGBlock(nn.Module):
    default_act = nn.ReLU()

    def __init__(self, c1, c2, deploy=False, act=True):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.deploy = deploy
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(self.c1, self.c2, 3, 1, 1, bias=True)
        else:
            self.rbr_3x3 = conv_bn(self.c1, self.c2, 3)
            self.rbr_1x1 = conv_bn(self.c1, self.c2, 1)

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_1x1'):
            kernel, bias = self.get_equivalent_kernel_bias()

            self.rbr_reparam = nn.Conv2d(self.c1, self.c2, 3, 1, 1, bias=True)
            self.rbr_reparam.weight.data = kernel
            self.rbr_reparam.bias.data = bias

            for para in self.parameters():
                para.detach_()

            # self.rbr_3x3 = self.rbr_reparam
            self.__delattr__('rbr_3x3')
            self.__delattr__('rbr_1x1')

            self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_3x3)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1


    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return torch.zeros(1), torch.zeros(1)
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            return torch.zeros(1), torch.zeros(1)
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - (running_mean * gamma / std)

    def forward(self, x):
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        else:
            return self.act(self.rbr_3x3(x) + self.rbr_1x1(x))

class REPADetectHead(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # self.cv2 = nn.ModuleList(
        #     nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        # )
        self.cv2 = nn.ModuleList(nn.Sequential(RepVGGBlock(x, c2), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(RepVGGBlock(x, c3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    # Use this version when exporting onnx
    '''
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW

        y = []
        for i in range(self.nl):
            y.append(self.cv2[i](x[i]))
            cls = torch.sigmoid(self.cv3[i](x[i]))
            cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
            y.append(cls)
            y.append(cls_sum)
        return y
    '''

    # Use this version when training model
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)

    def switch_to_deploy(self):
        for module_list in [self.cv2, self.cv3]:
            for module in module_list:
                for submodule in module:
                    if isinstance(submodule, RepVGGBlock):
                        submodule.switch_to_deploy()

class REPAPoseHead(REPADetectHead):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        # self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        self.cv4 = nn.ModuleList(nn.Sequential(RepVGGBlock(x, c4), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    # Use this version when exporting onnx
    '''
    def forward(self, x):
        bs = x[0].shape[0]  # batch size

        ps = []
        for i in range(self.nl):
            ps.append(self.cv4[i](x[i]))
        x = Detect_RepVGG.forward(self, x)
        # print(ps)
        return x, ps
    '''

    # Use this version when training model
    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = REPADetectHead.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y

    def switch_to_deploy(self):
        super().switch_to_deploy()
        for module in self.cv4:
            for submodule in module:
                if isinstance(submodule, RepVGGBlock):
                    submodule.switch_to_deploy()


if __name__ == "__main__":
    # Generating Sample image
    image1 = (1, 64, 32, 32)
    image2 = (1, 128, 16, 16)
    image3 = (1, 256, 8, 8)

    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image = [image1, image2, image3]
    channel = (64, 128, 256)
    # Model
    # mobilenet_v1 = Detect_RepVGG(nc=80, ch=channel)
    mobilenet_v1 = REPAPoseHead(nc=80, ch=channel)

    mobilenet_v1.eval()

    t1 = time.time()
    out1 = mobilenet_v1(image.copy())
    t2 = time.time()
    print(t2 - t1)

    mobilenet_v1.switch_to_deploy()

    t3 = time.time()
    out2 = mobilenet_v1(image.copy())
    t4 = time.time()
    print(t4 - t3)
