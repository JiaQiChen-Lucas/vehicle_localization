import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2
import math
import numpy as np
import time
import os
from typing import List, Tuple, Optional, Dict
# from .video_stream_online import MultiStreamManager, VideoStreamSimulator
from .video_stream_ffmpeg import MultiStreamManager, VideoStreamSimulator
from .log import Logger
from .yolov8_pose import letterbox
from .detect import detect_image
from .position import check_position_is_out, ImageResult, draw_image


def control_loop_frequency(start_time, target_interval=0.5):
    """
    控制频率
    :param start_time: 开始时间
    :param target_interval: 最小时间间隔
    :return:
    """
    elapsed = time.time() - start_time
    if elapsed < target_interval:
        time.sleep(target_interval - elapsed)


class ConcatenateImage:
    def __init__(self, shape: tuple = (640, 640, 3)):
        """
        初始化拼接图像对象

        参数:
            shape: 大图的尺寸，默认为 (640, 640, 3)
        """
        self.height, self.width, self.channels = shape
        self.concatenated_image = np.zeros(shape, dtype=np.uint8)
        self.patch_info: List[Dict] = []  # 记录每个拼接块的信息

    def add_image(self, frame: np.ndarray, row: int, col: int) -> bool:
        if row < 0 or row >= 2 or col < 0 or col >= 2:
            Logger.error(f"行列越界：row={row}, col={col}")
            return False

        if frame.size == 0:
            Logger.error(f"帧数据为空")
            return False

        try:
            # 缩放图像至 320x320 并保持比例
            img, padding, r = letterbox(
                frame,
                new_shape=(320, 320),
                color=(0, 0, 0),
                scaleup=True
            )

            # 拼接位置
            x_start = col * 320
            y_start = row * 320

            self.concatenated_image[y_start:y_start+320, x_start:x_start+320] = img

            # 记录元信息
            self.patch_info.append({
                'image': frame,
                'row': row,
                'col': col,
                'padded_shape': img.shape,
                'padding': padding,
                'scale_ratio': r
            })

            return True
        except Exception as e:
            Logger.error(f"插入图像出错：{e}")
            return False
def concatenate_images(frame_list: List[np.ndarray]) -> Optional[ConcatenateImage]:
    if not frame_list or len(frame_list) == 0:
        # Logger.warning("需要至少1张图片")
        return None

    if len(frame_list) > 4:
        Logger.error("最多支持4张图片拼接")
        return None

    try:
        # 创建640x640的目标图像画布
        concatenateImage = ConcatenateImage()

        # 拼接逻辑：2x2布局
        for i, frame in enumerate(frame_list):
            row = i // 2  # 行索引
            col = i % 2  # 列索引

            success = concatenateImage.add_image(frame, row=row, col=col)
            if not success:
                Logger.error(f"拼接第 {i} 张图像失败")
                continue  # 或者 break 根据需求决定

        return concatenateImage
    except Exception as e:
        Logger.error(f"拼接图像出错: {e}")
        return None


def track_objects(image_box: np.ndarray, image_cls: np.ndarray, image_score: np.ndarray, image_kpts: np.ndarray, simulator: VideoStreamSimulator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    进行目标跟踪
    :param image_box: 检测框
    :param image_cls: 类别
    :param image_score: 得分
    :param image_kpts: 关键点
    :param simulator: 视频流操作器
    :return:
    """
    if len(image_box) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    tracked_results = simulator.bytetracker.update(boxes=image_box, object_classes=image_cls, scores=image_score, keypoints=image_kpts)

    # 构造输出结构
    boxes = []
    classes = []
    scores = []
    keypoints = []
    track_ids = []

    for result in tracked_results:
        # 解包数据
        x1, y1, x2, y2 = result[:4]
        track_id = int(result[4])
        score = float(result[5])
        cls = int(result[6])
        idx = int(result[7])

        # 提取对应的关键点（第8位开始是展平后的关键点）
        kpts_flat = result[8:]

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(cls)
        keypoints.append(kpts_flat.tolist())
        track_ids.append(track_id)

    image_boxes = np.array(boxes, dtype=np.float32)
    image_classes = np.array(classes, dtype=np.int32)
    image_scores = np.array(scores, dtype=np.float32)
    image_keypoints = np.array(keypoints, dtype=np.float32)  # shape=(N, K*3)
    image_track_ids = np.array(track_ids, dtype=np.int32)

    return image_boxes, image_classes, image_scores, image_keypoints, image_track_ids

def parse_and_track(concatenateImage: ConcatenateImage, result: dict, simulator: VideoStreamSimulator) -> List[ImageResult]:
    """
    解析拼接图像的推理结果，并将检测框和关键点映射回各原始图像坐标系

    参数:
        concatenateImage (ConcatenateImage): 包含拼接图像及其元信息的对象
        result (dict): 推理结果，包括 'boxes', 'classes', 'scores', 'keypoints'

    返回:
        List[Dict]: 检测结果列表，每个元素包含类别、置信度、调整后的框和关键点
    """
    # 初始化返回结构
    parsed_result = []

    if "boxes" not in result or "classes" not in result or "scores" not in result or "keypoints" not in result:
        Logger.error("推理结果缺少必要字段")
        return parsed_result

    # 遍历所有 patch 的元信息（记录了每个小图在拼接图中的位置）
    for info in concatenateImage.patch_info:
        row = info['row']
        col = info['col']
        padding = info['padding']
        scale_ratio = info['scale_ratio']
        image = info['image']

        # 计算左上角偏移量（320x320 图像在 640x640 中的位置）
        x_offset = col * 320
        y_offset = row * 320

        box_list = []
        score_list = []
        cls_list = []
        keypoints_list = []

        # 遍历所有检测目标
        for i in range(len(result["boxes"])):
            box = result["boxes"][i]
            cls = result["classes"][i]
            score = result["scores"][i]
            keypoints = result["keypoints"][i]

            # 提取 box 坐标
            x1, y1, x2, y2 = box

            # 判断是否属于当前 patch（即是否落在该 patch 的区域）
            if not (x1 >= x_offset and y1 >= y_offset and x2 <= x_offset + 320 and y2 <= y_offset + 320):
                continue  # 不属于本 patch，跳过

            # 映射回原始图像坐标系
            adjusted_box = [
                (x1 - x_offset - padding[2]) / scale_ratio,  # left
                (y1 - y_offset - padding[0]) / scale_ratio,  # top
                (x2 - x_offset - padding[2]) / scale_ratio,  # right
                (y2 - y_offset - padding[0]) / scale_ratio  # bottom
            ]

            adjusted_keypoints = []
            for j in range(0, len(keypoints), 3):
                kpt_x = (keypoints[j] - x_offset - padding[2]) / scale_ratio
                kpt_y = (keypoints[j + 1] - y_offset - padding[0]) / scale_ratio
                conf = keypoints[j + 2]
                adjusted_keypoints.extend([kpt_x, kpt_y, conf])

            box_list.append(adjusted_box)
            score_list.append(score)
            cls_list.append(cls)
            keypoints_list.append(adjusted_keypoints)


        image_box = np.array(box_list, dtype=np.float32)
        image_score = np.array(score_list, dtype=np.float32)
        image_cls = np.array(cls_list, dtype=np.int32)
        image_kpts = np.array(keypoints_list, dtype=np.float32)

        # 获取目标跟踪id
        image_boxes, image_classes, image_scores, image_keypoints, image_track_ids = track_objects(
            image_box=image_box,
            image_cls=image_cls,
            image_score=image_score,
            image_kpts=image_kpts,
            simulator=simulator
        )

        # print(f"Before Tracking: boxes: {image_box}, scores: {image_score}, classes: {image_cls}, keypoints: {image_kpts}")
        # print(f"After Tracking: boxes: {image_boxes}, scores: {image_scores}, classes: {image_classes}, keypoints: {image_keypoints}, track_ids: {image_track_ids}")

        # 添加到对应 stream_id 的结果中
        parsed_result.append(ImageResult(image=image, boxes=image_boxes, scores=image_scores, classes=image_classes, keypoints=image_keypoints, track_ids=image_track_ids))

    return parsed_result

def parse_result_is_empty(parsed_result: List[ImageResult]) -> bool:
    """
    判断 parsed_result 是否为空

    参数:
        parsed_result: List[ImageResult]: 检测结果对象列表

    返回:
        bool: True 表示结果为空，False 表示结果不为空
    """
    for image_result in parsed_result:
        if len(image_result.boxes) > 0:
            return False
    return True


def save_image(parsed_result: List[ImageResult], stream_id: str, KEYPOINTS_THRESH: float = 0.5, RESULT_DIR: str = "result/track", IS_LOGGING: bool = False):
    # 创建 result 目录
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 遍历解析结果并绘制
    for idx, image_result in enumerate(parsed_result):
        # 绘制检测结果
        output_image = draw_image(image_result, KEYPOINTS_THRESH)

        # 获取当前时间戳，取后6位
        timestamp = str(int(time.time() * 1000))[-6:]

        # 构造文件名
        filename = f"{stream_id}_track_{timestamp}_{idx}.jpg"
        save_path = os.path.join(RESULT_DIR, filename)

        # 保存图像
        cv2.imwrite(save_path, output_image)

        if IS_LOGGING:
            Logger.success(f"保存图像至：{save_path}")


def is_vehicle_static(image_result_queue: deque, position_change_threshold=10.0) -> bool:
    """
    判断目标是否静止（基于最近帧中最大检测框的位移）

    参数:
        image_result_queue (deque): 最近几帧的 ImageResult 数据
        position_change_threshold (float): 位移阈值

    返回:
        bool: True 表示静止，False 表示移动
    """
    # 如果队列的长度小于4，返回False
    if len(image_result_queue) < 4:
        return False

    # 计算每个检测框的中心点坐标
    boxes_centers = []
    for image_result in image_result_queue:
        boxes = image_result.boxes
        if len(boxes) == 0:
            # 没有检测框，加入 None 作为占位
            boxes_centers.append((-1, -1))
            continue

        # 找出面积最大的框
        max_area = 0
        max_box = None
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_box = box

        if max_box is not None:
            x1, y1, x2, y2 = max_box.astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            boxes_centers.append((center_x, center_y))
        else:
            # 理论上不会走到这里，因为已经确保 len(boxes) > 0
            boxes_centers.append((-1, -1))

    # 检查boxes_centers中是否存在连续四个中心点的位置变化小于position_change_threshold
    count = 0
    # 遍历[0, len(boxes_centers) - 2]
    for i in range(len(boxes_centers) - 1):
        current_center = boxes_centers[i]
        next_center = boxes_centers[i + 1]

        if current_center == (-1, -1) or next_center == (-1, -1):
            count = 0
            continue

        # 计算两个中心点之间的距离
        distance = math.sqrt((current_center[0] - next_center[0]) ** 2 + (current_center[1] - next_center[1]) ** 2)
        # 如果距离小于阈值，则计数加1
        if distance <= position_change_threshold:
            count += 1
        else:   # 否则，计数归零
            count = 0

        if count >= 3:
            return True

    return False

def get_result(image: np.ndarray, result: dict, simulator: VideoStreamSimulator) -> Optional[ImageResult]:
    """
    解析单张图片的推理结果，只关注检测框最大的目标，并进行目标跟踪
    :param result: 推理结果（json格式）
    :param simulator: 视频流模拟器
    :return:
    """
    if "boxes" not in result or "classes" not in result or "scores" not in result or "keypoints" not in result:
        Logger.error("推理结果缺少必要字段")
        return None

    boxes = np.array(result["boxes"], dtype=np.float32)
    classes = np.array(result["classes"], dtype=np.int32)
    scores = np.array(result["scores"], dtype=np.float32)
    keypoints = np.array(result["keypoints"], dtype=np.float32)

    # 获取目标跟踪id
    image_boxes, image_classes, image_scores, image_keypoints, image_track_ids = track_objects(
        image_box=boxes,
        image_cls=classes,
        image_score=scores,
        image_kpts=keypoints,
        simulator=simulator
    )

    if image_boxes is None or len(image_boxes) == 0:
        return None

    # 计算每个检测框的面积
    areas = (image_boxes[:, 2] - image_boxes[:, 0]) * (image_boxes[:, 3] - image_boxes[:, 1])

    # 找到面积最大的检测框索引
    max_area_idx = np.argmax(areas)

    # 提取最大目标的信息
    max_box = image_boxes[[max_area_idx]]
    max_class = image_classes[[max_area_idx]]
    max_score = image_scores[[max_area_idx]]
    max_keypoints = image_keypoints[[max_area_idx]]
    max_track_id = image_track_ids[[max_area_idx]]

    return ImageResult(
        image=image,
        boxes=max_box,
        classes=max_class,
        scores=max_score,
        keypoints=max_keypoints,
        track_ids=max_track_id
    )

def count_keypoints_valid(keypoints: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    统计每个目标中有效关键点的数量（置信度 > threshold）

    参数:
        keypoints (np.ndarray): 形状为 (N, K*3)，其中 N 为目标数量，K 为关键点数量
        threshold (float): 置信度阈值，默认为 0.5

    返回:
        np.ndarray: 形状为 (N,)，表示每个目标的有效关键点数量
    """
    # 将 keypoints reshape 为 (N, K, 3)，分别表示 x, y, confidence
    N, length = keypoints.shape
    K = length // 3
    keypoints_reshaped = keypoints.reshape(N, K, 3)

    # 判断每个关键点的置信度是否超过阈值
    valid_mask = keypoints_reshaped[:, :, 2] >= threshold  # shape: (N, K)

    # 统计每个目标的有效关键点数量
    valid_counts = valid_mask.sum(axis=1)  # shape: (N,)

    return valid_counts

def fetch_frame_from_stream(simulator: VideoStreamSimulator) -> Optional[np.ndarray]:
    """
    从指定视频流中获取最新帧
    :param stream_id: 视频流ID
    :param manager: 多流管理器
    :return: 包含stream_id和帧的FrameInfo对象或None
    """
    if not simulator:
        return None

    return simulator.get_frame()


def save_frame_rate(frame_rate: float, save_path: str):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "frame_rate": frame_rate    # 处理速率
    }
    # 确保文件路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 追加写入 JSON 行到文件
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + '\n')


def save_frame_handler_time(frame_handler_time: float, save_path: str):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "frame_handler_time": frame_handler_time    # 处理速率
    }
    # 确保文件路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 追加写入 JSON 行到文件
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + '\n')


def processing_video_stream(
        manage: MultiStreamManager,
        simulator: VideoStreamSimulator,
        THREAD_SAMPLE_INTERVAL: float = 0.5,
        KEYPOINTS_THRESH: float = 0.5,
        RESPONSE_TIME_FILE_PATH: str = "./result/response_time/response_time.jsonl",
        IMAGE_SAVE_DIR_PATH: str = "./result/detect",
        FRAME_RATE_SAVE_DIR_PATH: str = "./result/frame_rate_thread.jsonl",
        FRAME_HANDLER_TIME_FILE_PATH_THREAD: str = "./result/frame_handler_time_thread.jsonl",
):
    """
    单独处理某个视频流
    :param manage: 视频流管理器
    :param simulator: 视频流操作器
    :param THREAD_SAMPLE_INTERVAL: 采样间隔
    :param KEYPOINTS_THRESH: 关键点阈值
    :param RESPONSE_TIME_FILE_PATH: 响应时间存储文件位置
    :return:
    """
    if simulator is None:
        Logger.error("video_stream is None")
        return

    # 创建一个最大容量为6的队列
    image_result_queue = deque(maxlen=6)

    # 存储处理完的目标id
    processed_track_ids = set()

    # 创建线程池并保存为变量
    executor = ThreadPoolExecutor(max_workers=4)

    start_time = time.time()

    frame_count = 1
    frame_start_time = time.time()

    frame_total_handler_time = 0.0

    while not simulator.isDone:
        start_time = time.time()

        if frame_count >= 10:
            save_frame_rate(frame_count / (start_time - frame_start_time), FRAME_RATE_SAVE_DIR_PATH)
            save_frame_handler_time(frame_total_handler_time / frame_count, FRAME_HANDLER_TIME_FILE_PATH_THREAD)
            frame_count = 0
            frame_total_handler_time = 0.0
            frame_start_time = start_time

        image_list: List[np.ndarray] = []
        futures = []

        # 当帧队列不足4张图片时，等待
        while len(futures) < 4 and not simulator.isDone:
            start_time = time.time()

            # 提交任务到线程池
            future = executor.submit(fetch_frame_from_stream, simulator)
            futures.append(future)
            frame_count += 1

            if len(futures) < 4:
                frame_total_handler_time += time.time() - start_time
                # 控制帧率
                control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)

        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                image_list.append(result)

        if simulator.isDone:
            break

        # 拼接图片
        concatenateImage = concatenate_images(image_list)

        if concatenateImage is not None:
            # 调用推理模块，获取推理结果
            result = detect_image(concatenateImage.concatenated_image, file_path=RESPONSE_TIME_FILE_PATH)

            # 解析推理结果，并跟踪目标
            parsed_result = parse_and_track(concatenateImage, result, simulator)

            # 如果没有推理结果，说明连续4帧都没有目标出现，则停止线程
            if parsed_result is None or parse_result_is_empty(parsed_result):
                Logger.info(f"视频流 {simulator.stream_id} 未解析到任何检测结果")
                frame_total_handler_time += time.time() - start_time
                # 控制帧率
                control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                break

            # 绘制并保存图片
            # save_image(parsed_result, simulator.stream_id, KEYPOINTS_THRESH)

            # 将推理结果加入队列中
            image_result_queue.extend(parsed_result)

            # 判断目标是否静止（可能一张图片内有多个目标，只关注检测框最大的那个）
            if is_vehicle_static(image_result_queue):
                # 清空队列
                image_result_queue.clear()

                Logger.info(f"视频流 {simulator.stream_id} 目标已静止")

                max_detect_count = 3

                # 最多读取 max_detect_count 检验
                while max_detect_count > 0:
                    start_time = time.time()

                    max_detect_count -= 1
                    # 读取一帧
                    current_frame = simulator.get_frame()
                    frame_count += 1

                    if current_frame is None:
                        Logger.error(f"视频流 {simulator.stream_id} 未读取到任何帧")
                        frame_total_handler_time += time.time() - start_time
                        # 控制帧率
                        control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                        continue

                    # 推理
                    result = detect_image(current_frame, file_path=RESPONSE_TIME_FILE_PATH)

                    # 解析推理结果
                    target_info = get_result(current_frame, result, simulator)

                    if target_info is None or target_info.boxes.shape[0] == 0:
                        Logger.info(f"视频流 {simulator.stream_id} 未检测到目标")
                        frame_total_handler_time += time.time() - start_time
                        # 控制帧率
                        control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                        continue

                    # 如果track_id已经被检测过
                    if target_info.track_ids[0] in processed_track_ids:
                        Logger.info(f"视频流 {simulator.stream_id} 目标已处理过")
                        break

                    # 统计有效关键点数量
                    valid_counts = count_keypoints_valid(target_info.keypoints, KEYPOINTS_THRESH)

                    if valid_counts[0] < 3:
                        Logger.info(f"视频流 {simulator.stream_id} 目标关键点数量不足")
                        frame_total_handler_time += time.time() - start_time
                        # 控制帧率
                        control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                        continue

                    # 判断是否越界
                    if check_position_is_out(image_result=target_info, stream_id=simulator.stream_id, RESULT_DIR=IMAGE_SAVE_DIR_PATH, is_save=False, KEYPOINTS_THRESH=KEYPOINTS_THRESH):
                        Logger.info(f"视频流 {simulator.stream_id} 目标越界")
                        frame_total_handler_time += time.time() - start_time
                        # 控制帧率
                        control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                        continue

                    # # 绘制并保存图片
                    # save_image(
                    #     parsed_result=[target_info],
                    #     stream_id=simulator.stream_id,
                    #     KEYPOINTS_THRESH=KEYPOINTS_THRESH,
                    #     RESULT_DIR=f"{IMAGE_SAVE_DIR_PATH}/success",
                    #     IS_LOGGING=True
                    # )

                    # 加入processed_track_ids
                    processed_track_ids.add(target_info.track_ids[0])
                    Logger.success(f"{simulator.stream_id} 目标已处理完毕，track_id: {target_info.track_ids}, box: {target_info.boxes}, cls: {target_info.classes}, score: {target_info.scores}, keypoints: {target_info.keypoints}")

                    # 退出内循环
                    break

        frame_total_handler_time += time.time() - start_time
        # 控制帧率
        control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)

    Logger.success(f"视频流 {simulator.stream_id} 处理完毕")

    # 设置该视频流的处理标志为False
    manage.set_processing_flag(stream_id=simulator.stream_id, contain_object=False)

