import numpy as np
from ultralytics import YOLO
import os
import torch
import cv2
from scipy.optimize import linear_sum_assignment
import shutil
import matplotlib.pyplot as plt
import pandas as pd

def calculate_diagonal_length(boxes_xywh):
    """
    计算矩形框的对角线长度
    :param boxes_xywh: 包含矩形框信息的numpy数组，每行表示一个矩形框的中心点坐标和宽度、高度
    :return: 对角线长度的numpy数组
    """
    diagonal_lengths = []
    for box in boxes_xywh:
        # 提取矩形框的中心点坐标、宽度、高度
        x, y, w, h = box[1:5]

        # 计算矩形框的对角线长度
        diagonal_length = np.sqrt(w ** 2 + h ** 2)
        diagonal_lengths.append(diagonal_length)

    return np.array(diagonal_lengths)

def calculate_area(boxes_xywh):
    """
    计算矩形框的面积
    :param boxes_xywh: 包含矩形框信息的numpy数组，每行表示一个矩形框的中心点坐标和宽度、高度
    :return: 矩形框面积的numpy数组
    """
    areas = []
    for box in boxes_xywh:
        # 提取矩形框的中心点坐标、宽度、高度
        x, y, w, h = box[1:5]

        # 计算矩形框的对角线长度
        area = np.sqrt(w ** 2 + h ** 2)
        areas.append(area)

    return np.array(areas)

# 根据阈值，确定预测的关键点是否存在
def determine_keypoint_visibility(pred_keypoints, threshold):
    keypoints_visibility = pred_keypoints[:, :, 2] >= threshold
    return keypoints_visibility

# 计算IoU（两个边界框重叠部分的面积/两个边界框的总面积减去重叠部分的面积，取值范围为[0,1]）
def calculate_iou(box1, box2, eps=1e-7):
    # box = [class_id, x_center, y_center, width, height]
    x1_min, y1_min = box1[1] - box1[3] / 2, box1[2] - box1[4] / 2
    x1_max, y1_max = box1[1] + box1[3] / 2, box1[2] + box1[4] / 2
    x2_min, y2_min = box2[1] - box2[3] / 2, box2[2] - box2[4] / 2
    x2_max, y2_max = box2[1] + box2[3] / 2, box2[2] + box2[4] / 2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + eps)

    return iou

def compute_metrics(pred_boxes, gt_boxes, pred_keypoints, gt_keypoints, class_num, pred_boxes_conf, box_iou_threshold=0.5, visibility_threshold=0.5, threshold_ratio=0.1, sigma=0.025):
    """
    计算指标
    :param pred_boxes: 预测框的numpy数组，形状为 (num_pred, 5)，包含 class_id, x_center, y_center, width, height
    :param gt_boxes: 真实框的numpy数组，形状为 (num_gt, 5)，包含 class_id, x_center, y_center, width, height
    :param pred_keypoints: 预测关键点的numpy数组，形状为 (num_pred, num_keypoints, 3)，包含 x, y, visibility
    :param gt_keypoints: 真实关键点的numpy数组，形状为 (num_gt, num_keypoints, 3)，包含 x, y, visibility
    :param class_num: 检测类别数量
    """
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    class_correct_keypoints = {i: 0 for i in range(class_num)}  # 记录每个类别预测正确的关键点的数量
    class_total_keypoints = {i: 0 for i in range(class_num)}    # 记录每个类别真实关键点的数量
    class_correct_boxes = {i: 0 for i in range(class_num)}      # 记录每个类别预测正确的检测框的数量
    class_total_boxes = {i: 0 for i in range(class_num)}        # 记录每个类别真实检测框的数量

    # oks
    class_total_oks = {i: 0 for i in range(class_num)}  # 记录每个类别真实检测框对应的oks总和

    # ap
    class_tp = {i: [] for i in range(class_num)}  # 1代表预测正例且实际正例，0代表预测正例且实际负例

    for i in range(num_gt):
        cls = gt_boxes[i][0]
        class_total_boxes[cls] += 1
        for j in range(gt_keypoints[i].shape[0]):
            if gt_keypoints[i][j][2] == 2:
                class_total_keypoints[cls] += 1

    # 构建代价矩阵，计算每个预测框和每个真实框之间的距离
    # 初始化为非常大的值，表示高代价
    cost_matrix = np.full((num_pred, num_gt), fill_value=np.inf)

    for i in range(num_pred):
        for j in range(num_gt):
            iou = calculate_iou(pred_boxes[i], gt_boxes[j])
            # 如果 IoU 不满足阈值 或 类别错误，设置一个高代价而不是 inf
            cost_matrix[i, j] = -iou if pred_boxes[i][0] == gt_boxes[j][0] and iou >= box_iou_threshold else 1e6

    # 使用匈牙利算法找到最佳匹配
    # row_ind 和 col_ind 是两个数组，它们分别表示最佳匹配的行索引（预测值）和列索引（标签值）
    # 在评估过程中，多余的预测（即没有匹配到真实标签的预测）通常不算入NME中。我们只关注那些成功匹配到真实标签的预测。
    # 在计算归一化平均误差（NME）时，我们只对成功匹配的预测和真实标签计算NME
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    new_row_ind = []
    new_col_ind = []

    for i, j in zip(row_ind, col_ind):
        if -cost_matrix[i, j] >= box_iou_threshold and pred_boxes[i][0] == gt_boxes[j][0]:  # 匹配有效
            new_row_ind.append(i)
            new_col_ind.append(j)
            cls = gt_boxes[j][0]
            class_correct_boxes[cls] += 1

    for i in range(num_pred):
        conf = pred_boxes_conf[i]
        if i in new_row_ind:
            class_tp[pred_boxes[i][0]].append((1, conf))
        else:
            class_tp[pred_boxes[i][0]].append((0, conf))

    matched_pred_keypoints = pred_keypoints[new_row_ind]    # 匹配成功后的检测框对应的关键点（预测值）
    matched_gt_keypoints = gt_keypoints[new_col_ind]    # 匹配成功后的检测框对应的关键点（真实值）
    matched_gt_boxes = gt_boxes[new_col_ind]        # 匹配成功后的检测框（真实值）
    matched_diag_lengths = calculate_diagonal_length(matched_gt_boxes)  # 匹配成功后的检测框对应的对角线长度（真实值）

    # oks
    matched_areas = calculate_area(matched_gt_boxes)  # 匹配成功后的检测框对应的面积（真实值）

    # 真实框与预测框匹配成功时
    for i in range(len(matched_pred_keypoints)):
        cls = matched_gt_boxes[i][0]
        total_val = 0
        gt_point_vis_count = 0
        for j in range(matched_pred_keypoints.shape[1]):
            pred_point = matched_pred_keypoints[i, j]
            gt_point = matched_gt_keypoints[i, j]

            # 真实值关键点不可见但预测值存在：直接忽略该预测点，不计算其误差。可以认为模型不应预测这个关键点，因此其误差计算应忽略这个点。
            if gt_point[2] == 0:
                continue  # Skip if ground truth point is not present

            gt_point_vis_count += 1

            # 可见性分数低于阈值，被认为是不存在的。
            # 真实值关键点存在但预测值不存在：这种情况表示模型没有成功预测出需要的关键点，这时可以将其误差设置为一个较大值（例如使用框对角线长度），以反映模型的错误。
            if pred_point[2] < visibility_threshold:
                continue
            else:
                # pck
                # 计算欧氏距离
                error = np.sqrt((pred_point[0] - gt_point[0]) ** 2 + (pred_point[1] - gt_point[1]) ** 2)
                # 归一化误差
                normalized_error = error / matched_diag_lengths[i]
                # print(f"第{i + 1}个预测框第{j + 1}个关键点的误差为{normalized_error}")
                if normalized_error <= threshold_ratio:      # 归一化误差小于等于阈值时，认为该关键点成功预测
                    class_correct_keypoints[cls] += 1

                # oks
                d = (pred_point[0] - gt_point[0]) ** 2 + (pred_point[1] - gt_point[1]) ** 2
                e = d / (2 * (sigma ** 2) * matched_areas[i])
                total_val += np.exp(-e)

        # 计算该匹配的 OKS 值，避免除以0
        oks = total_val / gt_point_vis_count if gt_point_vis_count > 0 else 0
        class_total_oks[cls] += oks

    return class_correct_keypoints, class_total_keypoints, class_correct_boxes, class_total_boxes, class_total_oks, class_tp

def load_label_boxes_and_keypoints(label_src, img_width, img_height):
    gt_boxes = []
    gt_keypoints = []
    with open(label_src, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = list(map(float, line.strip().split()))

            gt_boxes.append([parts[0], parts[1] * img_width, parts[2] * img_height, parts[3] * img_width, parts[4] * img_height])

            keypoints = []
            for i in range(5, len(parts), 3):
                x = parts[i] * img_width
                y = parts[i + 1] * img_height
                v = int(parts[i + 2])  # visibility
                keypoints.append([x, y, v])
            gt_keypoints.append(keypoints)

    return np.array(gt_boxes), np.array(gt_keypoints)

def add_dictionaries(dict1, dict2):
    result = dict1.copy()  # 复制 dict1 的内容到 result
    for key, value in dict2.items():
        if key in result:
            result[key] += value  # 如果键存在，累加值
        else:
            result[key] = value   # 如果键不存在，直接添加
    return result

def calculate_weighted_pck(class_correct_keypoints, class_total_keypoints, dataset_class_total_boxes):
    total_weighted_pck = 0
    total_boxes = sum(dataset_class_total_boxes.values())  # 计算所有类别的总检测框数量

    # 遍历所有类别
    for cls in class_total_keypoints:
        correct_kpts = class_correct_keypoints.get(cls, 0)  # 获取该类别的正确关键点数
        total_kpts = class_total_keypoints.get(cls, 0)  # 获取该类别的总关键点数

        if total_kpts > 0:  # 避免除以零
            pck_class = correct_kpts / total_kpts  # 计算该类别的PCK
            cls_boxes = dataset_class_total_boxes.get(cls, 0)
            weight = cls_boxes / total_boxes
            total_weighted_pck += weight * pck_class

    return total_weighted_pck

def calculate_weighted_oks(dataset_class_total_oks, dataset_class_total_boxes):
    total_weighted_oks = 0
    total_boxes = sum(dataset_class_total_boxes.values())  # 计算所有类别的总检测框数量

    if total_boxes == 0:
        return 1

    for cls in dataset_class_total_oks:
        cls_boxes = dataset_class_total_boxes.get(cls, 0)
        cls_oks = dataset_class_total_oks.get(cls, 0)

        if cls_boxes > 0:  # 如果该类别有检测框
            # 计算该类别的平均 OKS
            class_avg_oks = cls_oks / cls_boxes
            # 加权 OKS：按类别检测框数量比例进行加权
            weight = cls_boxes / total_boxes
            total_weighted_oks += class_avg_oks * weight

    return total_weighted_oks

def update_array(dataset_class_array, class_val):
    for cls_id, values in class_val.items():
        dataset_class_array[cls_id].extend(values)  # 使用 extend 将列表内容追加到 dataset_class_tp

    return dataset_class_array

def compute_ap(precision, recall, method="interp"):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        precision (list): The precision curve.
        recall (list): The recall curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    # method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def calculate_map50(dataset_class_tp, dataset_class_total_boxes, eps=1e-16):
    ap_array = []

    for cls in dataset_class_tp:
        cls_tp = dataset_class_tp.get(cls, [])
        cls_tp_sorted = sorted(cls_tp, key=lambda x: x[1], reverse=True)
        sorted_tp = np.array([x[0] for x in cls_tp_sorted])

        fpc = (1 - sorted_tp).cumsum(0)
        tpc = sorted_tp.cumsum(0)

        total_gt = dataset_class_total_boxes.get(cls, 0)

        precision = tpc / (tpc + fpc)
        recall = tpc / (total_gt + eps)

        cls_ap = compute_ap(precision=precision, recall=recall, method="interp")
        ap_array.append(cls_ap)

    return sum(ap_array) / len(ap_array)

def validate_model(weight_path, input_image_dir, input_label_dir, conf=0.8, iou_threshold=0.5, visibility_threshold=0.5, threshold_ratio=0.1, sigma=0.025):
    model = YOLO(weight_path)

    # 获取类别
    objs_labels = model.names  # get class labels
    class_num = len(objs_labels)
    # print(objs_labels)

    image_files = os.listdir(input_image_dir)

    total_time = 0

    dataset_class_correct_keypoints = {i: 0 for i in range(class_num)}  # 记录整个数据集中每个类别预测正确的关键点的数量
    dataset_class_total_keypoints = {i: 0 for i in range(class_num)}    # 记录整个数据集中每个类别真实关键点的数量
    dataset_class_correct_boxes = {i: 0 for i in range(class_num)}      # 记录整个数据集中每个类别预测正确的检测框的数量
    dataset_class_total_boxes = {i: 0 for i in range(class_num)}        # 记录整个数据集中每个类别真实检测框的数量

    dataset_class_total_oks = {i: 0 for i in range(class_num)}  # 记录整个数据集中每个类别真实检测框对应的oks总和

    dataset_class_tp = {i: [] for i in range(class_num)}

    for image_file in image_files:
        # print(image_file)
        # 读取图像
        image_path = os.path.join(input_image_dir, image_file)
        image = cv2.imread(image_path)

        result = model(image, verbose=False, conf=conf)[0]

        total_time += result.speed["inference"]

        boxes = result.boxes.cpu().numpy()  # Boxes object for bbox outputs
        # print(boxes)
        boxes_xywh = boxes.xywh
        # print(boxes_xywh)
        boxes_cls = boxes.cls
        # print(boxes_cls)
        pred_boxes_conf = boxes.conf
        # print(boxes.conf)

        # 合并class_id和box信息
        # pred_boxes包含 class_id, x_center, y_center, width, height
        pred_boxes = np.concatenate((boxes_cls.reshape(-1, 1), boxes_xywh), axis=1)
        # print(pred_boxes)

        keypoints = result.keypoints.cpu().numpy()
        # print(keypoints)
        # keypoints_xy = keypoints.xy
        # print(keypoints_xy)
        pred_keypoints = keypoints.data
        # print(pred_keypoints)

        label_filename = os.path.splitext(image_file)[0] + '.txt'
        label_src = os.path.join(input_label_dir, label_filename)

        # 获取图像的宽和高
        img_height, img_width = image.shape[:2]

        gt_boxes, gt_keypoints = load_label_boxes_and_keypoints(label_src, img_width, img_height)

        class_correct_keypoints, class_total_keypoints, class_correct_boxes, class_total_boxes, class_total_oks, class_tp = compute_metrics(pred_boxes=pred_boxes, gt_boxes=gt_boxes, pred_keypoints=pred_keypoints, gt_keypoints=gt_keypoints, class_num=class_num, pred_boxes_conf=pred_boxes_conf,box_iou_threshold=iou_threshold, visibility_threshold=visibility_threshold, threshold_ratio=threshold_ratio, sigma=sigma)

        dataset_class_correct_keypoints = add_dictionaries(dataset_class_correct_keypoints, class_correct_keypoints)
        dataset_class_total_keypoints = add_dictionaries(dataset_class_total_keypoints, class_total_keypoints)
        dataset_class_correct_boxes = add_dictionaries(dataset_class_correct_boxes, class_correct_boxes)
        dataset_class_total_boxes = add_dictionaries(dataset_class_total_boxes, class_total_boxes)
        dataset_class_total_oks = add_dictionaries(dataset_class_total_oks, class_total_oks)

        dataset_class_tp = update_array(dataset_class_tp, class_tp)

    box_map50 = calculate_map50(dataset_class_tp=dataset_class_tp, dataset_class_total_boxes=dataset_class_total_boxes)

    weighted_pck = calculate_weighted_pck(dataset_class_correct_keypoints, dataset_class_total_keypoints, dataset_class_total_boxes)
    weighted_oks = calculate_weighted_oks(dataset_class_total_oks, dataset_class_total_boxes)

    return weighted_pck, weighted_oks, box_map50

def get_pt_files(weight_dir):
    pt_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]
    return pt_files

if __name__ == '__main__':
    # 输入目录路径
    input_image_dir = '/media/hardDisk1/lucas/code/Wheel-YOLOv8-pose/datasets/vehicle_yolo/images/val'
    input_label_dir = '/media/hardDisk1/lucas/code/Wheel-YOLOv8-pose/datasets/vehicle_yolo/labels/val'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.device_count()

    root = os.getcwd()

    weight_dir = '/media/hardDisk1/lucas/code/Wheel-YOLOv8-pose/ultralytics-8.2.15/result/Wheel-YOLOv8-pose/ReLU/train/weights'

    saved_name = 'best_pck_oks.pt'

    weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]

    best_score = 0
    best_model = None

    # PCK 是衡量模型预测的关键点与真实关键点之间的准确度的指标。它表示在给定的阈值范围内，预测的关键点有多少个与真实关键点匹配。
    # OKS 是一种更为细致的评估指标，考虑了每个关键点的可见性和与真实关键点的距离。它基于每个关键点的预测得分，计算与真实关键点的相似度。

    epoch_scores = {}  # 用于存储epoch和对应的score

    for weight_file in weight_files:
        if weight_file == saved_name:
            continue
        elif 'epoch' in weight_file:
            epoch_number = int(weight_file.split('epoch')[1].split('.')[0])
            # print(epoch_number)
            if epoch_number < 0 or epoch_number > 800:
                continue
        elif weight_file == 'best.pt':
            continue
        elif weight_file == 'last.pt':
            epoch_number = 800
        else:
            continue

        weight_path = os.path.join(weight_dir, weight_file)
        weighted_pck, weighted_oks, box_map50 = validate_model(weight_path=weight_path, input_image_dir=input_image_dir, input_label_dir=input_label_dir, conf=0.8, iou_threshold=0.5, visibility_threshold=0.5, threshold_ratio=0.025, sigma=0.5)
        score = weighted_pck * 0.5 + weighted_oks * 0.4 + box_map50 * 0.1
        print(f"model: {weight_file}, weighted_pck: {weighted_pck}, weighted_oks: {weighted_oks}, box_map50: {box_map50}, score: {score}")
        if score > best_score:
            best_score = score
            best_model = weight_file

        # 记录当前epoch和score
        epoch_scores[epoch_number] = score

    print(f"best model on val-dataset: {best_model}, score: {best_score}")

    if best_model is not None:
        src_path = os.path.join(weight_dir, best_model)
        dest_path = os.path.join(weight_dir, saved_name)
        shutil.copy(src_path, dest_path)
        print(f"Best model {best_model} copied to {dest_path}")

    # 按照epoch排序
    sorted_epochs = sorted(epoch_scores.keys())
    sorted_scores = [epoch_scores[epoch] for epoch in sorted_epochs]

    # 绘制score曲线
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_epochs, sorted_scores, marker='o', color='b', label='Score', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Score over Epochs')
    plt.legend()
    plt.grid()

    # 保存图像到weight_dir目录下
    plot_file_path = os.path.join(weight_dir, 'curve.png')
    plt.savefig(plot_file_path)

    # 关闭图像，释放内存
    plt.close()
    print(f'Score curve saved at: {plot_file_path}')

    # 创建 DataFrame 用于保存到 Excel
    df = pd.DataFrame({
        'Epoch': sorted_epochs,
        'Score': sorted_scores
    })

    excel_path = os.path.join(weight_dir, 'epoch_scores.xlsx')

    # 将 DataFrame 保存到 Excel 文件
    df.to_excel(excel_path, index=False)

    print(f'Excel saved at: {excel_path}')