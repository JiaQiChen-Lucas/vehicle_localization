import os
import time
from typing import Tuple, List, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt
from .yolov8_pose import CLASSES, keypoint_classes
from .log import Logger

WEIGHT_BRIDGE_SIZE = (7, 14)
CORNER = [(939, 821), (1391, 850), (47, 920), (614, 1038)]

class ImageResult:
    def __init__(self, image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, keypoints: np.ndarray, track_ids: np.ndarray):
        self.image = image
        self.boxes = boxes
        self.scores = scores
        self.classes = classes
        self.keypoints = keypoints
        self.track_ids = track_ids

def draw_image(image_result: ImageResult, KEYPOINTS_THRESH: float = 0.5):
    image = image_result.image.copy()  # 避免修改原始图像

    if len(image_result.boxes) == 0:
        return image  # 没有检测结果，直接返回原图

    for i in range(len(image_result.boxes)):
        box = image_result.boxes[i]
        score = image_result.scores[i]
        cl = image_result.classes[i]

        top, left, right, bottom = box.astype(int)

        # 绘制检测框
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)

        # 绘制类别和得分
        label = f"{CLASSES[cl]} {score:.2f}"
        cv2.putText(image, label,
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # 如果存在关键点
        if image_result.keypoints is not None and len(image_result.keypoints) > i:
            keypoints = image_result.keypoints[i]

            for j in range(0, len(keypoints), 3):
                x = keypoints[j]
                y = keypoints[j + 1]
                conf = keypoints[j + 2]

                if conf > KEYPOINTS_THRESH:
                    idx = j // 3
                    x_int, y_int = int(x), int(y)
                    cv2.circle(image, (x_int, y_int), 3, (0, 255, 0), -1)

                    # 绘制关键点标签
                    if idx < len(keypoint_classes):
                        kp_label = f"{keypoint_classes[idx]}: {conf:.2f}"
                        cv2.putText(image, kp_label,
                                    (x_int, y_int - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 255, 0), 1)

        # 如果存在跟踪 ID
        if image_result.track_ids is not None and i < len(image_result.track_ids):
            track_id = image_result.track_ids[i]
            cv2.putText(image, f"ID:{track_id}",
                        (top, bottom + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    return image

def compute_homography(
        weightbridge_size: Tuple[int, int],
        corner: List[Tuple[int, int]],
        use_ransac: bool = False
) -> Optional[np.ndarray]:
    """
    计算单应性矩阵 H，用于将图像坐标映射到实际物理坐标。

    参数:
        weightbridge_size (Tuple[int, int]): 地磅的实际尺寸 (width, height)
        corner (List[Tuple[int, int]]): 图像中地磅的四个角点坐标 [(x1,y1), (x2,y2), ...]
        use_ransac (bool): 是否使用 RANSAC 提高鲁棒性，默认为 False

    返回:
        Optional[np.ndarray]: 3x3 的单应性矩阵 H，计算失败返回 None
    """
    if len(corner) < 4:
        print("Error: 至少需要4个角点来计算单应性矩阵")
        return None

    width, height = weightbridge_size

    # 图像中的源点
    src_points = np.array(corner[:4], dtype=np.float32)

    # 实际物理坐标系的目标点
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ], dtype=np.float32)

    # 使用 getPerspectiveTransform 或 findHomography
    try:
        if not use_ransac:
            H = cv2.getPerspectiveTransform(src_points, dst_points)
        else:
            H, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        return H
    except Exception as e:
        print(f"计算单应性矩阵时出错: {e}")
        return None


def transform_keypoints(keypoints, H, KEYPOINTS_THRESH: float = 0.5):
    transformed_keypoints = []

    # 处理关键点
    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i + 1]
        kp_conf = keypoints[i + 2]

        # 仅处理置信度大于阈值的关键点
        if kp_conf > KEYPOINTS_THRESH:
            # 将关键点 (x, y) 转换为齐次坐标 (x, y, 1)
            point = np.array([x, y, 1.0])

            # 使用单应性矩阵 H 进行坐标转换
            transformed_point = np.dot(H, point)

            # 将结果除以齐次坐标的第三个分量，恢复到普通的二维坐标
            transformed_point /= transformed_point[2]

            # 提取转换后的 x, y
            new_x, new_y = transformed_point[0], transformed_point[1]

            # 将转换后的关键点添加到结果列表
            transformed_keypoints.append([new_x, new_y])

    return np.array(transformed_keypoints)


# 计算两个向量的点积
def dot_product(v1, v2):
    return np.dot(v1, v2)


def angle_closeness_to_right_angle(A, B, C):
    # 计算向量 AB 和 BC
    AB = B - A
    BC = C - B
    # 计算点积
    dot_prod = dot_product(AB, BC)
    return abs(dot_prod)  # 返回点积的绝对值，越接近0表示角越接近90度


def adjust_keypoints(points):
    '''
    调整预测点
    '''
    # 如果能够直接预测到4个点，则只需要进行矫正，无需估算出第4个点的位置
    if len(points) == 4:
        return points

    A = points[0]
    B = points[1]
    C = points[2]

    # 计算每个角的接近直角的程度
    angle_ABC = angle_closeness_to_right_angle(A, B, C)
    angle_BCA = angle_closeness_to_right_angle(B, C, A)
    angle_CAB = angle_closeness_to_right_angle(C, A, B)

    # 输出哪个角最接近直角
    angles = [angle_ABC, angle_BCA, angle_CAB]
    min_angle_index = np.argmin(angles)  # 找到最小的点积，即最接近直角的角

    d = np.sum((points - np.roll(points, -1, axis=0)) ** 2, axis=1)

    # 显示最接近直角的角
    if min_angle_index == 0:  # 角 ABC 最接近直角
        if d[0] > d[1]:  # C投影到BA上
            # 计算点C在直线AB上的投影
            adjust_points = [A, B, C]
        else:  # A投影到BC上
            adjust_points = [C, B, A]
    elif min_angle_index == 1:  # 角 BCA 最接近直角
        if d[1] > d[2]:  # A投影到CB上
            adjust_points = [B, C, A]
        else:  # B投影到CA上
            adjust_points = [A, C, B]
    else:  # 角 CAB 最接近直角
        if d[2] > d[0]:  # B投影到AC上
            adjust_points = [C, A, B]
        else:  # C投影到AB上
            adjust_points = [B, A, C]

    projection_length = np.dot(adjust_points[2] - adjust_points[0], adjust_points[1] - adjust_points[0]) / np.dot(
        adjust_points[1] - adjust_points[0], adjust_points[1] - adjust_points[0])
    projection_vector = projection_length * (adjust_points[1] - adjust_points[0])

    adjust_points[1] = adjust_points[0] + projection_vector

    adjust_points.append(adjust_points[0] + adjust_points[2] - adjust_points[1])

    return np.array(adjust_points, dtype=np.float32)


def is_outside_of_weightbridge(
        points: np.ndarray,
        weightbridge_size: Tuple[int, int] = WEIGHT_BRIDGE_SIZE,
        tolerance: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
) -> bool:
    """
    判断点是否超出地磅边界，支持不同方向设置不同误差容忍度。

    参数:
        points (np.ndarray): 坐标点数组，形状为 (N, 2)
        weightbridge_size (Tuple[int, int]): 地磅尺寸 (width, height)
        tolerance Tuple[float, float, float, float]: 四个方向的误差
    返回:
        bool: 如果有任何一点超出带容忍度的边界，返回 True；否则返回 False
    """

    # 输入检查
    if not isinstance(points, np.ndarray) or len(points.shape) != 2 or points.shape[1] < 2:
        raise ValueError("points 必须是形状为 (N, 2) 的 numpy 数组")

    width, height = weightbridge_size
    left_tol, right_tol, top_tol, bottom_tol = tolerance

    for point in points:
        x, y = point[:2]
        if (
                x < -left_tol or
                x > width + right_tol or
                y < -bottom_tol or
                y > height + top_tol
        ):
            return True
    return False


def draw_weightbridge(img: np.ndarray, corner: List[Tuple[int, int]]):
    # 提取四个角的坐标
    top_left = tuple(corner[0])
    top_right = tuple(corner[1])
    bottom_left = tuple(corner[2])
    bottom_right = tuple(corner[3])

    # 画出地磅的边界框（连线形成矩形）
    cv2.line(img, top_left, top_right, (0, 255, 0), 2)  # 上边
    cv2.line(img, top_right, bottom_right, (0, 255, 0), 2)  # 右边
    cv2.line(img, bottom_right, bottom_left, (0, 255, 0), 2)  # 下边
    cv2.line(img, bottom_left, top_left, (0, 255, 0), 2)  # 左边

    # 在四个角上标记点
    cv2.circle(img, top_left, 5, (255, 0, 0), -1)  # 左上角
    cv2.circle(img, top_right, 5, (255, 0, 0), -1)  # 右上角
    cv2.circle(img, bottom_left, 5, (255, 0, 0), -1)  # 左下角
    cv2.circle(img, bottom_right, 5, (255, 0, 0), -1)  # 右下角


def draw_BEV(weightbridge_size: Tuple[int, int], keypoints: np.ndarray, adjust_points: np.ndarray, save_path: str):
    plt.figure(figsize=(10, 6))

    # 提取地磅实际尺寸
    width, height = weightbridge_size

    # 定义地磅在实际坐标系中的目标点
    weightbridge_coordinates = np.array([
        [0, 0],  # 实际左上角点
        [width, 0],  # 实际右上角点
        [0, height],  # 实际左下角点
        [width, height]  # 实际右下角点
    ], dtype=np.float32)

    plt.plot([weightbridge_coordinates[0, 0], weightbridge_coordinates[1, 0], weightbridge_coordinates[3, 0],
              weightbridge_coordinates[2, 0], weightbridge_coordinates[0, 0]],
             [weightbridge_coordinates[0, 1], weightbridge_coordinates[1, 1], weightbridge_coordinates[3, 1],
              weightbridge_coordinates[2, 1], weightbridge_coordinates[0, 1]],
             'b-', label="Weighbridge")
    plt.scatter(weightbridge_coordinates[:, 0], weightbridge_coordinates[:, 1], color='blue',
                label="Keypoints of Weighbridge")

    if adjust_points is not None and adjust_points.size > 0:
        plt.plot([adjust_points[0, 0], adjust_points[1, 0], adjust_points[2, 0],
                  adjust_points[3, 0], adjust_points[0, 0]],
                 [adjust_points[0, 1], adjust_points[1, 1], adjust_points[2, 1],
                  adjust_points[3, 1], adjust_points[0, 1]],
                 'y-', label="Vehicle-Gt-Adjust")
        plt.scatter(adjust_points[:, 0], adjust_points[:, 1], color='yellow',
                    label="Keypoints of Vehicle-Gt-Adjust")

    plt.scatter(keypoints[:, 0],
                keypoints[:, 1],
                color='red',
                label="Keypoints of Vehicle-Pred")

    plt.title("BEV")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis('equal')  # 强制保持比例尺一致
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()  # 反转y轴以符合图像坐标系

    plt.savefig(save_path, format="svg", bbox_inches="tight")

    plt.close()


def check_position_is_out(
        image_result: ImageResult,
        stream_id: str,
        RESULT_DIR: str = "./result",
        is_save: bool = True,
        KEYPOINTS_THRESH: float = 0.5,
        weightbridge_size: Tuple[int, int] = WEIGHT_BRIDGE_SIZE,
        corner: List[Tuple[int, int]] = CORNER
):
    """
    判断是否越界
    """
    # 计算单应性矩阵
    H = compute_homography(weightbridge_size, corner)

    # 使用单应性矩阵转换关键点
    transformed_keypoints = transform_keypoints(keypoints=image_result.keypoints[0], H=H,
                                                KEYPOINTS_THRESH=KEYPOINTS_THRESH)

    if len(transformed_keypoints) < 3 or len(transformed_keypoints) > 5:
        return False

    # 矫正关键点，并预测第四个点
    adjust_points = adjust_keypoints(transformed_keypoints)

    if is_save:
        # 创建 result 目录
        os.makedirs(RESULT_DIR, exist_ok=True)

        image = draw_image(image_result=image_result, KEYPOINTS_THRESH=KEYPOINTS_THRESH)
        draw_weightbridge(image, corner)

        # 获取当前时间戳，取后6位
        timestamp = str(int(time.time() * 1000))[-6:]

        # 构造文件名
        filename = f"{stream_id}_{timestamp}_position.jpg"
        save_path = os.path.join(RESULT_DIR, filename)

        # 保存图像
        cv2.imwrite(save_path, image)
        Logger.success(f"视频流 {stream_id} 已检测是否越界，保存图像至：{save_path}")

        bev_filename = f"{stream_id}_{timestamp}_bev.svg"
        bev_save_path = os.path.join(RESULT_DIR, bev_filename)
        draw_BEV(weightbridge_size, transformed_keypoints, adjust_points, bev_save_path)

    return is_outside_of_weightbridge(adjust_points, weightbridge_size)


if __name__ == '__main__':
    H = compute_homography(WEIGHT_BRIDGE_SIZE, CORNER)
    print(H)
