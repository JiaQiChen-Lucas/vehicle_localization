import cv2
import numpy as np
from rknnlite.api import RKNNLite
import threading
from .log import Logger
from typing import List, Optional
from typing import Tuple
from queue import Queue, Empty
from concurrent.futures import Future
from .yolov8_pose import InferImage

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
MAX_DETECT = 300

# The follew two param is for mAP test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, seg_part):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part


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


def class_agnostic_nms(boxes, scores, nms_thresh, max_detections=MAX_DETECT):
    """
    Class-agnostic NMS implementation using numpy.

    :param boxes: np.ndarray (N, 4), format [x1, y1, x2, y2]
    :param scores: np.ndarray (N,)
    :param nms_thresh: float, IoU threshold
    :param max_detections: int, 最大保留检测框数量
    :return: list of kept indices
    """
    if len(boxes) == 0:
        return []

    # 混合坐标和得分排序
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0 and len(keep) < max_detections:
        i = order[0]
        keep.append(i)

        # 计算当前框与其他框的交并比
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1 + 1e-5)
        h = np.maximum(0.0, yy2 - yy1 + 1e-5)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 1]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        ovr = inter / (area_i + area_order - inter)

        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def _crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        masks: np.ndarray of shape (N, H, W)
        boxes: np.ndarray of shape (N, 4), format [x1, y1, x2, y2]

    Returns:
        np.ndarray of shape (N, H, W)
    """
    n, h, w = masks.shape

    # 提取坐标并转换为整数像素坐标
    x1 = boxes[:, 0].reshape(-1, 1, 1)
    y1 = boxes[:, 1].reshape(-1, 1, 1)
    x2 = boxes[:, 2].reshape(-1, 1, 1)
    y2 = boxes[:, 3].reshape(-1, 1, 1)

    # 创建网格坐标矩阵
    r = np.arange(w).reshape(1, 1, w)  # shape: (1, 1, W)
    c = np.arange(h).reshape(1, h, 1)  # shape: (1, H, 1)

    # 构建布尔掩码：判断是否在框内
    mask_x = (r >= x1) & (r < x2)  # shape: (N, 1, W)
    mask_y = (c >= y1) & (c < y2)  # shape: (N, H, 1)

    # 合并成最终掩码：(N, H, W)
    final_mask = mask_x & mask_y

    # 应用到原始 masks 上
    return masks * final_mask.astype(masks.dtype)


def resize_mask(mask, size):
    """
    Resize a set of binary masks using bilinear interpolation.

    Args:
        mask: np.ndarray of shape (N, H, W)
        size: tuple (width, height)

    Returns:
        resized_masks: np.ndarray of shape (N, height, width)
    """
    resized_masks = []
    for m in mask:
        m_resized = cv2.resize(m, size, interpolation=cv2.INTER_LINEAR)
        resized_masks.append(m_resized)
    return np.array(resized_masks)


def post_process(input_data):
    """
    处理 YOLOv8-seg 模型输出，包括检测框、类别置信度、得分和分割掩码。
    :param input_data: 模型推理结果
    :return: 处理后的边界框、类别、得分和分割掩码
    """
    proto = input_data[-1]  # proto 用于恢复分割掩码
    boxes, scores, classes_conf, seg_parts = [], [], [], []
    default_branch = 3  # 默认分支数
    pair_per_branch = len(input_data) // default_branch  # 每个分支包含的数据量

    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
        seg_parts.append(input_data[pair_per_branch * i + 3])

    def sp_flatten(_in):
        ch = _in.shape[1]  # 获取通道数
        _in = _in.transpose(0, 2, 3, 1)  # 调整维度顺序
        return _in.reshape(-1, ch)  # 展平为二维数组

    # 对各个部分进行展平操作
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_parts = [sp_flatten(_v) for _v in seg_parts]

    # 将展平后的数据连接成完整的数组
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_parts = np.concatenate(seg_parts)

    # 根据阈值过滤
    boxes, classes, scores, seg_parts = filter_boxes(boxes, scores, classes_conf, seg_parts)

    zipped = zip(boxes, classes, scores, seg_parts)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)

    max_nms = 30000
    n = boxes.shape[0]  # number of boxes
    if not n:
        return np.array([]), np.array([]), np.array([]), np.array([])
    elif n > max_nms:  # excess boxes
        boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_part = [np.array(x) for x in result]

    # nms
    nboxes, nclasses, nscores, nseg_part = [], [], [], []
    real_keeps = class_agnostic_nms(boxes, scores, NMS_THRESH, MAX_DETECT)
    nboxes.append(boxes[real_keeps])
    nclasses.append(classes[real_keeps])
    nscores.append(scores[real_keeps])
    nseg_part.append(seg_part[real_keeps])

    if not nclasses and not nscores:
        return np.array([]), np.array([]), np.array([]), np.array([])

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_part = np.concatenate(nseg_part)

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_part.shape[-1], -1)
    seg_img = np.matmul(seg_part, proto)
    seg_img = sigmoid(seg_img)
    seg_img = seg_img.reshape(-1, ph, pw)

    seg_threadhold = 0.5

    # crop seg outside box
    # Resize mask to 640x640
    seg_img = resize_mask(seg_img, (640, 640))

    # Crop mask outside bounding boxes
    seg_img = _crop_mask(seg_img, boxes)

    seg_img = seg_img > seg_threadhold
    return boxes, classes, scores, seg_img


def draw(image, boxes, scores, classes, seg_masks):
    """
    在图像上绘制检测框和分割掩码
    :param image: 输入图像
    :param boxes: 检测框坐标
    :param scores: 检测框得分
    :param classes: 检测框类别
    :param seg_masks: 分割掩码
    """
    overlay = np.zeros_like(image, dtype=np.uint8)
    for box, score, cl, mask in zip(boxes, scores, classes, seg_masks):
        # 将检测框的坐标解包为顶部、左侧、右侧和底部坐标
        top, left, right, bottom = box

        # 打印当前检测框的类别和对应的得分
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        # 打印当前检测框的坐标信息
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
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

        # Resize mask to match image size
        # resized_mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # 绘制分割掩码（叠加在原图上）
        color = np.random.randint(0, 255, (1, 3)).tolist()[0]  # 随机颜色
        mask_color = np.zeros_like(image, dtype=np.uint8)
        mask_color[mask > 0] = color  # 将掩码区域涂上颜色
        overlay = cv2.add(overlay, mask_color)

    image = cv2.addWeighted(image, 1, overlay, 0.5, 0)

    return image



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    '''
    调整图像大小，填充两边
    :param img: 原图像
    :param new_shape: 调整后图像的大小
    :param color: 填充颜色，(114, 114, 114)代表灰色
    :param scaleup: 是否允许放大图像
    :return: 处理后的图片
    '''
    shape = img.shape[:2]   # 获取输入图像的高度和宽度
    if isinstance(new_shape, int):  # 如果 new_shape 是一个整数，则将其转换为 (new_shape, new_shape) 的形式，即将图像调整为一个正方形
        new_shape = (new_shape, new_shape)
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


def adjust_boxes(boxes, padding, r):
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


def adjust_seg(seg, padding, image_shape):
    """
    根据图像处理时的 padding 和缩放比例，将分割掩码还原到原始图像尺寸。

    :param seg: 分割掩码 (N, H, W)
    :param padding: (top, bottom, left, right)
    :param r: 缩放比例
    :return: 还原后的分割掩码 (N, H_orig, W_orig)
    """
    # 如果没有检测框，直接返回空数组
    if seg.size == 0 or seg.shape[0] == 0:
        return seg

    top, bottom, left, right = padding
    h, w = seg.shape[1], seg.shape[2]
    orig_h, orig_w = image_shape

    seg = seg[:, top:h-bottom, left:w-right]

    # 转换为 (H, W, N) 以便 resize
    seg = np.where(seg, 1, 0).astype(np.uint8).transpose(1, 2, 0)
    # Resize 到原始图像尺寸，并保持通道维度
    if seg.shape[2] == 1:
        seg = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]
    else:
        seg = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # 再转回 (N, H, W)
    seg = seg.transpose(2, 0, 1)

    return seg


def load_rknn(RKNN_MODEL):
    rknn_flag = True

    # 加载 RKNN 模型
    rknn = RKNNLite()
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        rknn_flag = False
        Logger.error('Load RKNN model failed!')
    else:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            rknn_flag = False
            Logger.error('Init runtime environment failed!')
        else:
            Logger.success(f"RKNN model ({RKNN_MODEL}) initialized successfully!")

    return rknn, rknn_flag

# 后处理
# def decode_seg(outputs: List[np.ndarray], inferImage: InferImage):
#     boxes, classes, scores, seg_masks = post_process(outputs)
#
#     real_box = adjust_boxes(boxes, inferImage.padding, inferImage.r)
#     real_seg = adjust_seg(seg_masks, inferImage.padding, inferImage.image_shape[:2])
#
#     return {
#         "boxes": real_box.tolist() if real_box is not None else None,
#         "classes": classes.tolist() if classes is not None else None,
#         "scores": scores.tolist() if scores is not None else None,
#         "seg_masks": real_seg.tolist() if real_seg is not None else None
#     }

def decode_seg(outputs: List[np.ndarray], inferImage: InferImage):
    boxes, classes, scores, seg_masks = post_process(outputs)

    return {
        "image_shape:": inferImage.image_shape,
        "model_input_shape": inferImage.input_data.shape,
        "padding": inferImage.padding,
        "r": inferImage.r,
        "boxes": boxes.tolist() if boxes is not None else None,
        "classes": classes.tolist() if classes is not None else None,
        "scores": scores.tolist() if scores is not None else None,
        "seg_masks": seg_masks.tolist() if seg_masks is not None else None,
    }

def detectFrame(rknn, inferImage: InferImage):
    # 进行推理
    return rknn.inference(inputs=[inferImage.input_data])


class Yolov8SegExecutor:
    def __init__(self, rknn_model: str, max_workers: int = 3, cores: int = 3):
        self.rknn_model = rknn_model
        self.max_workers = max_workers
        self.cores = cores

        self.task_queue = Queue()
        self.rknn_instances = self._init_rknn_instances(rknn_model, max_workers, cores)
        self.threads = []
        self._active = True

        for i in range(max_workers):
            t = threading.Thread(target=self._worker, args=(i,), name=f"RKNNWorker-{i}")
            t.daemon = True
            t.start()
            self.threads.append(t)

    def _init_rknn(self, rknnModel: str, id: int = 0) -> Tuple[Optional[RKNNLite], bool]:
        """初始化单个RKNN实例"""
        rknn = RKNNLite()
        try:
            ret = rknn.load_rknn(rknnModel)
            if ret != 0:
                Logger.error(f'Load RKNN model failed with error code: {ret}')
                rknn.release()
                return None, False

            if id == 0:
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
            elif id == 1:
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
            elif id == 2:
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
            elif id == -1:
                ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            else:
                ret = rknn.init_runtime()

            if ret != 0:
                Logger.error(f'Init runtime failed on core {id}, error: {ret}')
                rknn.release()
                return None, False

            Logger.success(f"RKNN initialized (Core: {id}, Model: {rknnModel})")
            return rknn, True
        except Exception as e:
            Logger.error(f"RKNN initialization exception: {str(e)}")
            rknn.release()
            return None, False

    def _init_rknn_instances(self, rknnModel: str, max_workers: int = 1, cores: int = 3) -> List[RKNNLite]:
        """
        初始化多个RKNN实例
        :param rknnModel: RKNN模型路径
        :param max_workers: 需要初始化的实例数量
        :param cores: NPU核心数
        :return: 成功初始化的rknn实例列表
        :raises: RuntimeError 如果没有任何实例初始化成功
        """
        rknn_list = []
        success_count = 0

        for i in range(max_workers):
            core_id = i % cores
            rknn, success = self._init_rknn(rknnModel, core_id)
            if success:
                rknn_list.append(rknn)
                success_count += 1
            else:
                Logger.warning(f"Failed to initialize instance {i} on core {core_id}")

        if success_count == 0:
            error_msg = "All RKNN instances failed to initialize!"
            Logger.error(error_msg)
            raise RuntimeError(error_msg)

        Logger.success(f"Successfully initialized {success_count}/{max_workers} RKNN instances")
        return rknn_list

    def _worker(self, instance_id):
        rknn = self.rknn_instances[instance_id]

        while self._active:
            try:
                input_data, future = self.task_queue.get(timeout=1)  # 阻塞获取任务
                try:
                    result = detectFrame(rknn=rknn, inferImage=input_data)
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
        self._active = False
        for t in self.threads:
            t.join(timeout=2)
        for i, rknn in enumerate(self.rknn_instances):
            try:
                rknn.release()
                Logger.success(f"RKNN instance {i} released")
            except Exception as e:
                Logger.error(f"Failed to release RKNN instance {i}: {str(e)}")

