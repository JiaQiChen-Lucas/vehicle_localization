import time
from queue import Queue, Empty

import cv2
import numpy as np
from ais_bench.infer.interface import InferSession
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
from .log import Logger
from typing import List, Optional, Tuple
from typing import Tuple
from concurrent.futures import Future

OBJ_THRESH = 0.8
NMS_THRESH = 0.7

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("concrete_mixer_truck", "tanker", "truck")

id_list = [1, 2, 3]

keypoint_classes = ['lf', 'rf', 'lr', 'rr', 'lfi', 'rfi', 'lri', 'rri']


def filter_boxes(boxes, box_confidences, box_class_probs, keypoints):
    """Filter boxes with object threshold.
    """
    '''
    boxes：表示检测框的位置信息，是一个形状为 (8400, 4) 的数组
    box_confidences：初始的检测框的置信度，是一个形状为 (8400,) 的数组，其值全部为1
    box_class_probs：表示每个检测框的类别概率，是一个形状为 (8400, C) 的数组，其中 8400 表示检测框的数量，C 表示类别的数量，每个值表示对应检测框属于各个类别的概率
    keypoints：表示关键点的信息，是一个形状为 (8400, 3 * pc) 的数组，pc为关键点的数量，3表示每个关键点有 x, y, conf
    '''

    # 对box_confidences进行了形状重塑，将其变成了一个一维数组
    box_confidences = box_confidences.reshape(-1)
    # candidate表示检测框的数量，class_num表示类别的数量
    # candidate, class_num = box_class_probs.shape

    # 计算每个检测框的最大类别概率和对应的类别
    # axis=-1 表示在最后一个维度上计算最大值。对于形状为 (N, C) 的数组，这意味着在每个检测框的所有类别概率中寻找最大值。
    class_max_score = np.max(box_class_probs, axis=-1)  # class_max_score 是一个形状为 (N,) 的一维数组，其中每个元素表示对应检测框的最大类别概率。
    # np.argmax 是 NumPy 中用于寻找最大值的索引的函数
    # axis=-1 表示在最后一个维度上寻找最大值的索引。对于形状为 (N, C) 的数组，这意味着在每个检测框的所有类别概率中寻找概率最大的类别索引
    classes = np.argmax(box_class_probs, axis=-1)   # classes 是一个形状为 (N,) 的一维数组，其中每个元素表示对应检测框的最大概率类别的索引。

    # 通过乘积操作得到每个检测框的得分（即置信度乘以最大类别概率），并筛选出得分大于等于 OBJ_THRESH 的检测框
    # np.where 是 NumPy 的一个函数，用于返回满足条件的元素的索引
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)

    # 根据筛选出的索引，从 boxes、classes、scores和keypoints 中获取相应的数据
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    keypoints = keypoints[_class_pos]

    return boxes, classes, scores, keypoints


# 非极大值抑制
def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    # 将 boxes 数组中的每个检测框的坐标和宽高提取出来
    x = boxes[:, 0]     # 左上角x
    y = boxes[:, 1]     # 左上角y
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h  # 计算每个检测框的面积，用于后续计算重叠面积
    order = scores.argsort()[::-1]  # 根据得分对检测框进行排序，并返回得分从高到低的索引

    keep = []  # 初始化一个空列表，用于存储保留的检测框索引
    while order.size > 0:
        i = order[0]  # 取出当前得分最高的检测框索引
        keep.append(i)  # 将当前检测框索引加入到保留列表中

        # 计算当前检测框与其余检测框的交集部分的坐标
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        # 计算交集部分的宽度和高度
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        # 计算交集的面积
        inter = w1 * h1

        # 计算交并比
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 筛选出交并比小于等于NMS_THRESH的检测框索引
        inds = np.where(ovr <= NMS_THRESH)[0]
        # 更新待处理的检测框索引，将已处理的检测框移除
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def dfl(position):
    '''
    将模型预测的特征图上的位置分布转换为实际的连续偏移量。
    :param position: 模型预测的特征图上的位置分布
    :return: 每个子项（x1, y1, x2, y2）相对于中心点（特征图上每个像素）的实际连续偏移量
    '''
    # position.shape可能为(1,64,80,80)、(1,64,40,40)、(1,64,20,20)
    # 以position.shape为(1,64,80,80)为例
    x = np.array(position)  # 将输入的位置信息 position 转换为 NumPy 数组，以便后续处理
    n, c, h, w = x.shape  # 获取输入数组的形状信息，其中n表示样本数量，c表示通道数，h和w分别表示高度和宽度
    p_num = 4  # 这里假设每个位置有 4 个子项（通常表示边界框的 x1, y1, x2, y2）
    mc = c // p_num  # 每个位置信息的通道数量，代表16种偏移量，偏移量被离散化为 0 到 15 这 16 个值
    # 将输入数组按照定义好的形状进行重新排列，以便后续对每个子项进行处理
    y = x.reshape(n, p_num, mc, h, w)  # y的维度为(1, 4, 16, 80, 80)，包含每个偏移量的概率

    # Softmax along the channel dimension (axis=2)
    # 首先对每个子项进行指数运算，这样可以保留相对大小关系，但会使数值变得更大或更小
    '''
    np.max(y, axis=2, keepdims=True) 计算每个子项的最大值，用于数值稳定性。
    np.exp(y - np.max(y, axis=2, keepdims=True)) 计算指数。
    exp_y / np.sum(exp_y, axis=2, keepdims=True) 对指数进行归一化，得到概率分布。
    '''
    exp_y = np.exp(y - np.max(y, axis=2, keepdims=True))  # 维度为(1, 4, 16, 80, 80)
    # 然后对每个子项的指数结果进行归一化，使用 Softmax 函数将其转换为概率分布，使得每个子项的值在0到1之间，并且所有子项的概率之和为1
    y = exp_y / np.sum(exp_y, axis=2, keepdims=True)  # 维度为(1, 4, 16, 80, 80)

    # Create an accumulation matrix
    # 创建了一个累积矩阵，用于后续的位置信息组合
    '''
    累积矩阵的主要作用是与 softmax 归一化后的概率分布进行元素级的乘法运算，从而实现加权平均。这一步的核心在于将离散的概率分布转换为连续的坐标。
    np.arange(mc) 生成一个从 0 到 mc-1 的数组，表示离散位置的索引。
    astype(float) 将这些索引转换为浮点数，以便后续计算。
    reshape(1, 1, mc, 1, 1) 将数组的形状调整为 (1, 1, mc, 1, 1)，使其能够与归一化后的概率分布相乘
    '''
    # 累积矩阵 [0, 1, 2, ..., 15] 表示离散位置的索引值，有 0 到 15 这 16 个偏移量
    acc_matrix = np.arange(mc).astype(float).reshape(1, 1, mc, 1, 1)  # 维度为(1, 1, 16, 1, 1)

    # Element-wise multiplication and summation along the channel dimension
    # y * acc_matrix 这一步将概率分布与相应的索引值相乘，乘法的结果是，每个位置的概率乘以其对应的索引值，得到加权的结果。
    # .sum(2) 对 mc 维度求和，得到加权平均的结果，即连续偏移量。
    # y * acc_matrix的维度为(1, 4, 16, 80, 80)
    y = (y * acc_matrix).sum(2)  # (1, 4, 80, 80)

    return y

# 将特征图上的预测位置被转换为输入图像上的实际边界框坐标
def box_process(position):
    # 获取了 position 的形状的第三个和第四个维度的大小，即特征图的高度和宽度。
    grid_h, grid_w = position.shape[2:4]
    # 使用 np.meshgrid() 函数创建了网格坐标
    # np.meshgrid 生成两个矩阵，col 和 row，表示特征图上每个位置的列和行坐标
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    # 将网格坐标的形状从 (grid_h, grid_w) 转换为 (1, 1, grid_h, grid_w)，以便后续计算
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    # 将横向和纵向的坐标连接在一起，得到了表示每个网格位置的中心坐标（即特征图上每个像素点的坐标）的 grid
    grid = np.concatenate((col, row), axis=1)   # 形状为(1, 2, grid_h, grid_w)
    # 计算了网格的步长，即特征图上每个单元在原始图像中所占的像素大小
    # stride 通过将输入图像尺寸除以特征图尺寸计算得到，并通过 reshape 调整形状为 (1, 2, 1, 1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    # 计算偏移量
    position = dfl(position)
    # 计算了边界框的左上角坐标 box_xy 和右下角坐标 box_xy2
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    # 将左上角坐标和右下角坐标乘以网格的步长，并将它们连接在一起，得到了表示真实边界框坐标的 xyxy 数组
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)  # 形状为(1, 4, grid_h, grid_w)

    return xyxy

def keypoints_process(position):
    # 获取了 position 的形状的第三个和第四个维度的大小，即特征图的高度和宽度。
    grid_h, grid_w = position.shape[2:4]
    # 使用 np.meshgrid() 函数创建了网格坐标
    # np.meshgrid 生成两个矩阵，col 和 row，表示特征图上每个位置的列和行坐标
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    # 将网格坐标的形状从 (grid_h, grid_w) 转换为 (1, 1, grid_h, grid_w)，以便后续计算
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    # 将横向和纵向的坐标连接在一起，得到了表示每个网格位置的中心坐标（即特征图上每个像素点的坐标）的 grid
    grid = np.concatenate((col, row), axis=1)  # 形状为(1, 2, grid_h, grid_w)
    # 计算了网格的步长，即特征图上每个单元在原始图像中所占的像素大小
    # stride 通过将输入图像尺寸除以特征图尺寸计算得到，并通过 reshape 调整形状为 (1, 2, 1, 1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position[:, 0::3] = (position[:, 0::3] * 2.0 + grid[:, :1]) * stride[:, :1]
    position[:, 1::3] = (position[:, 1::3] * 2.0 + grid[:, 1:]) * stride[:, 1:]
    # 对置信度应用 Sigmoid 函数
    position[:, 2::3] = 1 / (1 + np.exp(-position[:, 2::3]))  # Sigmoid 函数

    return position


def post_process(input_data):
    # 前9个为原有的检测输出，后3个为关键点检测输出
    box_outputs = input_data[:9]
    keypoints_outputs = input_data[9:]

    boxes, scores, classes_conf, keypoints = [], [], [], []
    default_branch = 3  # 默认分支数
    # 把输入数据的总长度除以默认的分支数，得到每个分支所包含的数据的数量
    # input_data的长度为9
    pair_per_branch = len(box_outputs) // default_branch
    # Python 忽略 score_sum 输出
    for i in range(default_branch):  # 循环遍历每个分支
        boxes.append(box_process(box_outputs[pair_per_branch * i]))
        classes_conf.append(box_outputs[pair_per_branch * i + 1])
        # 创建一个与当前分支类别置信度信息具有相同形状的数组，并且所有元素的值都被设置为1
        # np.ones_like() 函数会创建一个与输入数组具有相同形状的数组，并且数组中的所有元素都设置为1
        # [:, :1, :, :]表示保留所有的维度，但是只保留第二个维度的第一个元素（第二个维度变成了大小为1）
        # 这行代码的主要作用是为当前分支创建一个形状与类别置信度信息相同的数组，但只保留类别置信度信息的第一个通道，并将所有元素设置为 1，然后将这个数组添加到 scores 列表中。
        # 这种操作可能是为了在后续处理步骤中使用一个初始的置信度分数数组，该数组中的所有值都是 1。具体用途可能取决于后续代码的逻辑。
        scores.append(np.ones_like(box_outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        keypoints.append(keypoints_process(keypoints_outputs[i]))

    # 定义 sp_flatten(_in) 函数，用于展平数组
    def sp_flatten(_in):
        ch = _in.shape[1]  # 获取通道数
        # 将原来的维度顺序(B, C, H, W)转换成了(B, H, W, C)
        _in = _in.transpose(0, 2, 3, 1)
        # -1表示自动推断这个维度的大小
        # 展平后得到的数组的形状是(N, C)，其中N表示展平后的数组的长度
        return _in.reshape(-1, ch)

    # 对 boxes、classes_conf 和 scores 进行展平操
    # boxes、classes_conf、scores的长度都为3
    # 例：boxes[0]由(1,4,80,80)变换为(6400,4)
    boxes = [sp_flatten(_v) for _v in boxes]
    # 例：classes_conf[0]由(1,80,80,80)变换为(6400,80)
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    # 例：scores[0]由(1,1,80,80)变换为(6400,1)
    scores = [sp_flatten(_v) for _v in scores]
    # 例：scores[0]由(1,24,80,80)变换为(6400,24)。注：一共8个关键点
    keypoints = [sp_flatten(_v) for _v in keypoints]

    # 将展平后的数据进行连接操作，得到完整的边界框、类别置信度和得分数据
    boxes = np.concatenate(boxes)  # boxes的维度为(8400, 4)
    classes_conf = np.concatenate(classes_conf)  # 维度为(8400, 80)
    scores = np.concatenate(scores)  # 初始的检测框的置信度，其值全部为1，其维度为(8400, 1)
    keypoints = np.concatenate(keypoints)   # 维度为(8400, 24)

    # filter according to threshold
    # 返回的boxes, classes, scores的形状分别为(N, 4)、(N, )、(N, )，其中N为检测框的个数
    boxes, classes, scores, keypoints = filter_boxes(boxes, scores, classes_conf, keypoints)

    # nms
    # 初始化三个空列表，用于存储每个类别经过过滤后的检测框、对应的类别和得分
    nboxes, nclasses, nscores, nkeypoints = [], [], [], []
    # 遍历筛选后的检测框中，所有不同的类别
    for c in set(classes):
        # 获取当前类别 c 对应的检测框索引
        inds = np.where(classes == c)
        # 根据当前类别的索引，从原始的检测框、类别和得分数组中提取出当前类别的检测框、类别和得分
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        k = keypoints[inds]
        # 对当前类别的检测框和得分进行非极大值抑制处理，得到保留的检测框索引
        keep = nms_boxes(b, s)

        # 如果保留的检测框不为空
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
            nkeypoints.append(k[keep])

    # 检查是否没有保留的类别和得分，如果是，则表示没有检测到任何目标，直接返回空值
    if not nclasses and not nscores:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 将所有类别的检测框、类别和得分连接成一个数组，并更新为最终的检测结果
    boxes = np.concatenate(nboxes)      # 形状为(N, 4)
    classes = np.concatenate(nclasses)  # 形状为(N, )
    scores = np.concatenate(nscores)    # 形状为(N, )
    keypoints = np.concatenate(nkeypoints)  # 形状为(N, 3 * pc)，pc为关键点个数

    return boxes, classes, scores, keypoints


def draw(image, boxes, scores, classes, keypoints, KEYPOINTS_THRESH=0.5):
    # 使用 zip 函数同时迭代检测框、得分、类别和关键点，以便在每次迭代中处理一个检测结果
    for box, score, cl, keypoint in zip(boxes, scores, classes, keypoints):
        # 将检测框的坐标解包为顶部、左侧、右侧和底部坐标
        top, left, right, bottom = box

        # 打印当前检测框的类别和对应的得分
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # 打印当前检测框的坐标信息
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        # 将检测框的坐标转换为整数类型，因为绘图函数通常接受整数类型的坐标
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        # 在图像上绘制矩形框，参数分别为图像、左上角坐标、右下角坐标、颜色和线条粗细
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        # 在图像上添加文字，显示类别和置信度得分
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # 处理关键点
        for i in range(0, len(keypoint), 3):
            x = keypoint[i]
            y = keypoint[i + 1]
            kp_conf = keypoint[i + 2]

            # 仅绘制置信度大于阈值的关键点
            if kp_conf > KEYPOINTS_THRESH:
                index = int(i / 3)
                # 打印当前检测框的类别和对应的得分
                # print('keypoint: {}, x: {}, y: {}, conf: {}'.format(keypoint_classes[index], x, y, kp_conf))
                x = int(x)
                y = int(y)
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(image, '{}: {:.2f}'.format(keypoint_classes[index], kp_conf), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114), scaleup: bool = True) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
    """
    调整图像大小，保持纵横比并填充边缘。

    参数:
        img: 输入图像，形状为 (H, W, C)，C 通常为 3（BGR）
        new_shape: 目标尺寸，可以是整数（表示正方形）或 (height, width)
        color: 填充颜色 (BGR 格式)
        scaleup: 是否允许放大图像

    返回:
        img: 调整后的图像 (np.ndarray)
        padding: (top, bottom, left, right) 填充像素数
        r: 缩放比例
    """
    shape = img.shape[:2]   # 获取输入图像的高度和宽度

    # 如果当前图像尺寸已匹配目标尺寸，直接返回原图和默认参数
    if shape == (new_shape[0], new_shape[1]):
        return img, (0, 0, 0, 0), 1.0

    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])   # 计算缩放比例 r，确保图像在调整大小时不会变形，并且能完整地适应在新形状中
    if not scaleup:     # 如果 scaleup 为 False，则仅允许图像缩小，而不允许放大，以防止图像变得模糊
        r = min(r, 1.0)
    # 计算缩放后的图像尺寸 new_unpad，保持图像的纵横比
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    # 计算填充尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2
    dh /= 2
    # 将原图resize到new_unpad（长边相同，比例相同的新图）
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下两侧的padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算左右两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (top, bottom, left, right), r

def remove_letterbox(img: np.ndarray, padding: Tuple[int, int, int, int], r: float = 1.0, if_resize: bool = False) -> np.ndarray:
    """
    去掉图像的灰边，并恢复到原始大小

    参数:
        img: 输入图像，形状为 (H, W, C)
        padding: 灰边信息，格式为 (top, bottom, left, right)
        r: 缩放比例，用于还原尺寸
        if_resize: 是否需要将图像恢复为原始尺寸

    返回:
        处理后的图像
    """
    # 如果未缩放且无填充，直接返回原图
    if padding == (0, 0, 0, 0) and r == 1.0:
        return img

    top, bottom, left, right = padding
    # 去掉灰边
    img = img[top:img.shape[0]-bottom, left:img.shape[1]-right]

    if if_resize:
        # 获取去掉灰边后的图像尺寸
        shape = img.shape[:2]

        original_height = shape[0] / r
        original_width = shape[1] / r

        img = cv2.resize(img, (int(original_width), int(original_height)), interpolation=cv2.INTER_LINEAR)

    return img

def adjust_boxes(boxes: np.ndarray, padding: Tuple[int, int, int, int], r: float) -> np.ndarray:
    """
    根据图像处理时的 padding 和缩放比例，将检测框坐标还原到原始图像尺寸。

    参数:
        boxes (List[List[float]]): 检测框列表，每个检测框格式为 [x1, y1, x2, y2]
        padding (Tuple[int, int, int, int]): 图像填充信息，格式为 (top, bottom, left, right)
        r (float): 图像缩放比例，用于还原坐标

    返回:
        np.ndarray: 调整后的检测框数组，形状为 (N, 4)，其中 N 表示检测框数量
    """

    if padding == (0, 0, 0, 0) and r == 1.0:
        return boxes

    top, bottom, left, right = padding
    adjusted_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        # 去除 padding 偏移并还原缩放
        x1 = (x1 - left) / r
        y1 = (y1 - top) / r
        x2 = (x2 - left) / r
        y2 = (y2 - top) / r
        adjusted_boxes.append([x1, y1, x2, y2])

    return np.array(adjusted_boxes)

def adjust_keypoints(keypoints: np.ndarray, padding: Tuple[int, int, int, int], r: float) -> np.ndarray:
    """
    根据图像处理时的 padding 和缩放比例，将关键点坐标还原到原始图像尺寸。

    参数:
        keypoints (List[List[float]]): 关键点列表，每个关键点格式为 [x, y, conf] * K，K 表示关键点个数
        padding (Tuple[int, int, int, int]): 图像填充信息，格式为 (top, bottom, left, right)
        r (float): 图像缩放比例，用于还原坐标

    返回:
        np.ndarray: 调整后的关键点数组，形状为 (N, 3*K)，其中 N 表示检测目标数量，K 表示每个目标的关键点数量
    """
    if padding == (0, 0, 0, 0) and r == 1.0:
        return keypoints

    top, bottom, left, right = padding
    adjusted_keypoints = []

    for kp in keypoints:
        adjusted_kp = []
        for i in range(0, len(kp), 3):
            x = (kp[i] - left) / r
            y = (kp[i + 1] - top) / r
            conf = kp[i + 2]
            adjusted_kp.extend([x, y, conf])
        adjusted_keypoints.append(adjusted_kp)

    return np.array(adjusted_keypoints)

def convert_boxes_to_xywh(boxes):
    '''将boxes从x1, y1, x2, y2格式转换为x_center, y_center, w, h格式'''
    converted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        converted_boxes.append([x_center, y_center, w, h])
    return np.array(converted_boxes)


class InferImage:
    def __init__(self, input_data: np.ndarray, padding: Tuple[int, int, int, int], r: float):
        self.input_data = input_data
        self.padding = padding
        self.r = r

# 前处理
def image_pre_process(image: np.ndarray) -> InferImage:
    frame, padding, r = letterbox(image)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    return InferImage(input_data, padding, r)

# 后处理
def decode_result(outputs: List[np.ndarray], inferImage: InferImage):
    boxes, classes, scores, keypoints = post_process(outputs)

    boxes = adjust_boxes(boxes, inferImage.padding, inferImage.r)
    keypoints = adjust_keypoints(keypoints, inferImage.padding, inferImage.r)

    return {
        "boxes": boxes.tolist() if boxes is not None else None,
        "classes": classes.tolist() if classes is not None else None,
        "scores": scores.tolist() if scores is not None else None,
        "keypoints": keypoints.tolist() if keypoints is not None else None
    }

def detectFrame(cann, inferImage: InferImage):
    # 进行推理
    return cann.infer(feeds=[inferImage.input_data], mode="static")

# class CANNPoolExecutor:
#     def __init__(self, cann_model: str, max_workers: int = 6, cores: int = 3):
#         """
#         CANN线程池执行器
#
#         :param cann_model: CANN模型文件路径
#         :param func: 推理处理函数 (cann_instance, input_data) -> result
#         :param max_workers: 最大工作线程数
#         :param cores: NPU核心数
#         """
#         self.cann_model = cann_model
#         self.cann_instances = self._init_cann_instances(cann_model, max_workers, cores)
#         self.executor = ThreadPoolExecutor(max_workers=len(self.cann_instances), thread_name_prefix="CANNWorker")
#         self.lock = threading.Lock()
#         self.thread_num = 0    # 当前线程编号
#         self.workers = len(self.cann_instances)     # 总线程数
#         self._active = True
#
#     def _init_cann(self, cannModel: str, id: int = 0) -> Tuple[Optional[InferSession], bool]:
#         """初始化单个CANN实例"""
#         model = None
#         try:
#             if id == 0:
#                 model = InferSession(device_id=0, model_path=cannModel, )
#             elif id == 1:
#                 model = InferSession(device_id=0, model_path=cannModel)
#             elif id == 2:
#                 model = InferSession(device_id=0, model_path=cannModel)
#             elif id == 3:
#                 model = InferSession(device_id=0, model_path=cannModel)
#             else:
#                 model = InferSession(device_id=0, model_path=cannModel)
#
#             Logger.success(f"CANN initialized (Core: {id}, Model: {cannModel})")
#             return model, True
#         except Exception as e:
#             Logger.error(f"CANN initialization exception: {str(e)}")
#             return None, False
#
#     def _init_cann_instances(self, cannModel: str, max_workers: int = 1, cores: int = 4) -> List[InferSession]:
#         """
#         初始化多个CANN实例
#         :param rknnModel: RKNN模型路径
#         :param max_workers: 需要初始化的实例数量
#         :param cores: NPU核心数
#         :return: 成功初始化的rknn实例列表
#         :raises: RuntimeError 如果没有任何实例初始化成功
#         """
#         cann_list = []
#         success_count = 0
#
#         for i in range(max_workers):
#             core_id = i % cores
#             cann, success = self._init_cann(cannModel, core_id)
#             if success:
#                 cann_list.append(cann)
#                 success_count += 1
#             else:
#                 Logger.warning(f"Failed to initialize instance {i} on core {core_id}")
#
#         if success_count == 0:
#             error_msg = "All CANN instances failed to initialize!"
#             Logger.error(error_msg)
#             raise RuntimeError(error_msg)
#
#         Logger.success(f"Successfully initialized {success_count}/{max_workers} RKNN instances")
#         return cann_list
#
#     def submit(self, input_data):
#         """
#         提交推理任务
#
#         :param input_data: 输入数据
#         :param is_request: 是否为外部请求
#         :return: Future对象
#         :raises: RuntimeError 如果执行器已关闭
#         """
#         if not self._active:
#             raise RuntimeError("Executor has been shutdown")
#
#         def _task(instance_id: int):
#             try:
#                 cann = self.cann_instances[instance_id % len(self.cann_instances)]
#                 return detectFrame(cann=cann, frame=input_data)
#             except Exception as e:
#                 Logger.error(f"Task failed: {str(e)}")
#                 raise
#
#         with self.lock:
#             instance_id = self.thread_num
#             self.thread_num = (self.thread_num + 1) % self.workers
#
#         future = self.executor.submit(_task, instance_id)
#         return future
#
#     def shutdown(self):
#         """安全关闭执行器"""
#         if not self._active:
#             return
#
#         self._active = False
#         self.executor.shutdown(wait=True)
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.shutdown()



class CANNPoolExecutor:
    def __init__(self, cann_model: str, max_workers: int = 1, cores: int = 1):
        self.cann_model = cann_model
        self.max_workers = max_workers
        self.cores = cores

        self.task_queue = Queue()
        self.cann_instances = self._init_cann_instances(cann_model, max_workers, cores)
        self.threads = []
        self._active = True

        for i in range(max_workers):
            t = threading.Thread(target=self._worker, args=(i,), name=f"CANNWorker-{i}")
            t.daemon = True
            t.start()
            self.threads.append(t)

    def _init_cann(self, cannModel: str, id: int = 0) -> Tuple[Optional[InferSession], bool]:
        """初始化单个CANN实例"""
        model = None
        try:
            if id == 0:
                model = InferSession(device_id=0, model_path=cannModel, )
            elif id == 1:
                model = InferSession(device_id=0, model_path=cannModel)
            elif id == 2:
                model = InferSession(device_id=0, model_path=cannModel)
            elif id == 3:
                model = InferSession(device_id=0, model_path=cannModel)
            else:
                model = InferSession(device_id=0, model_path=cannModel)

            Logger.success(f"CANN initialized (Core: {id}, Model: {cannModel})")
            return model, True
        except Exception as e:
            Logger.error(f"CANN initialization exception: {str(e)}")
            return None, False

    def _init_cann_instances(self, cannModel: str, max_workers: int = 1, cores: int = 1) -> List[InferSession]:
        """
        初始化多个CANN实例
        :param rknnModel: RKNN模型路径
        :param max_workers: 需要初始化的实例数量
        :param cores: NPU核心数
        :return: 成功初始化的rknn实例列表
        :raises: RuntimeError 如果没有任何实例初始化成功
        """
        cann_list = []
        success_count = 0

        for i in range(max_workers):
            core_id = i % cores
            cann, success = self._init_cann(cannModel, core_id)
            if success:
                cann_list.append(cann)
                success_count += 1
            else:
                Logger.warning(f"Failed to initialize instance {i} on core {core_id}")

        if success_count == 0:
            error_msg = "All CANN instances failed to initialize!"
            Logger.error(error_msg)
            raise RuntimeError(error_msg)

        Logger.success(f"Successfully initialized {success_count}/{max_workers} RKNN instances")
        return cann_list

    def _worker(self, instance_id):
        cann = self.cann_instances[instance_id]

        while self._active:
            try:
                input_data, future = self.task_queue.get(timeout=1)  # 阻塞获取任务
                try:
                    result = detectFrame(cann=cann, inferImage=input_data)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.task_queue.task_done()
            except Empty:
                continue  # 超时重试
            except Exception as e:
                Logger.error(f"Worker-{instance_id} crashed: {e}")

    def submit(self, inferImage: InferImage):
        if not self._active:
            raise RuntimeError("Executor is shut down")

        future = Future()
        self.task_queue.put((inferImage, future))
        return future

    def shutdown(self):
        if not self._active:
            return

        self._active = False
        for t in self.threads:
            t.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


if __name__ == "__main__":
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "model", "best_Pose_RepVGG_ReLU_train_opset12_deploy.om")

    print(f"Loading model from: {model_path}")
    executor = CANNPoolExecutor(cann_model=model_path, max_workers=1, cores=1)

    image_path = "/home/HwHiAiUser/vehicle_yolov8/task_scheduling/app/result/stitched_20250617_174447.jpg"

    # 读取图片
    image = cv2.imread(image_path)

    future = executor.submit(image)

    result = future.result(timeout=0.5)

    print(result)
