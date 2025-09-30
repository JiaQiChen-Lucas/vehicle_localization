import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import time
import threading
from typing import Optional

import cv2
import numpy as np
# from .video_stream_online import MultiStreamManager, VideoStreamSimulator
from .video_stream_ffmpeg import MultiStreamManager, VideoStreamSimulator
from .image_process import FrameInfo, stitch_images, parse_inference_result
from .log import Logger
from .processing import processing_video_stream
from .detect import detect_image

KEYPOINTS_THRESH = 0.8


def control_loop_frequency(start_time, target_interval=0.5):
    """
    控制频率
    :param start_time: 开始时间
    :param target_interval: 最小时间间隔
    :return:
    """
    elapsed = time.time() - start_time

    if elapsed < target_interval:
        # Logger.info(f"sleep: {target_interval - elapsed}")
        time.sleep(target_interval - elapsed)
    # else:
        # Logger.info(f"超时: {elapsed - target_interval}")


def manager_is_closed(manager: MultiStreamManager) -> bool:
    """
    判断多流管理器是否已关闭
    :param manager: 多流管理器
    :return: True表示已关闭，False表示未关闭
    """
    for stream_id in manager.simulators:
        if not manager.get_stream(stream_id).isDone:
            return False
    return True


def create_response_time_file(response_time_dir_path: str = "./result/response_time"):
    # 创建目录（如果不存在）
    if not os.path.exists(response_time_dir_path):
        os.makedirs(response_time_dir_path)

    # 构造文件名（以当前时间命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_time_{timestamp}.jsonl"
    response_time_file_path = os.path.join(response_time_dir_path, filename)

    return response_time_file_path


def create_image_save_dir(image_save_prefix: str = "./result/image"):
    # 创建目录（如果不存在）
    if not os.path.exists(image_save_prefix):
        os.makedirs(image_save_prefix)

    # 构造路径名（以当前时间命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_dir = os.path.join(image_save_prefix, timestamp)

    os.makedirs(image_save_dir, exist_ok=True)  # 创建时间戳目录

    return image_save_dir


def create_frame_rate_save_dir(frame_rate_dir_path: str = "./result/frame_rate"):
    # 创建目录（如果不存在）
    if not os.path.exists(frame_rate_dir_path):
        os.makedirs(frame_rate_dir_path)

    # 构造文件名（以当前时间命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_main = f"frame_rate_{timestamp}_main.jsonl"
    filename_thread = f"frame_rate_{timestamp}_thread.jsonl"
    frame_rate_file_path_main = os.path.join(frame_rate_dir_path, filename_main)
    frame_rate_file_path_thread = os.path.join(frame_rate_dir_path, filename_thread)

    return frame_rate_file_path_main, frame_rate_file_path_thread


def create_image_time_save_dir(image_time_save_prefix: str = "./result/image_rate"):
    # 创建目录（如果不存在）
    if not os.path.exists(image_time_save_prefix):
        os.makedirs(image_time_save_prefix)

    # 构造路径名（以当前时间命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_rate_{timestamp}.jsonl"
    image_time_file_path = os.path.join(image_time_save_prefix, filename)

    return image_time_file_path

def create_frame_handler_time_file(frame_handler_time_path: str = "./result/frame_handler_time"):
    # 创建目录（如果不存在）
    if not os.path.exists(frame_handler_time_path):
        os.makedirs(frame_handler_time_path)

    # 构造文件名（以当前时间命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_main = f"frame_handler_time_{timestamp}_main.jsonl"
    filename_thread = f"frame_handler_time_{timestamp}_thread.jsonl"
    frame_handler_time_file_path_main = os.path.join(frame_handler_time_path, filename_main)
    frame_handler_time_file_path_thread = os.path.join(frame_handler_time_path, filename_thread)

    return frame_handler_time_file_path_main, frame_handler_time_file_path_thread


def fetch_frame_from_stream(stream_id, manager):
    """
    从指定视频流中获取最新帧
    :param stream_id: 视频流ID
    :param manager: 多流管理器
    :return: 包含stream_id和帧的FrameInfo对象或None
    """
    if manager.get_processing_flag(stream_id):
        return None

    simulator = manager.get_stream(stream_id)
    if not simulator:
        return None

    frame = simulator.get_frame()
    if frame is not None:
        return FrameInfo(stream_id, frame)

    return None


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


def stitch_handler_start(
        sleep_time: float = 10.0,
        SAMPLE_INTERVAL: float = 1.0,
        THREAD_SAMPLE_INTERVAL: float = 0.5,
        response_time_dir_path: str = "./result/response_time",
        image_save_prefix: str = "./result/image",
        frame_rate_save_prefix: str = "./result/frame_rate",
        image_time_save_prefix: str = "./result/image_rate",
        frame_handler_time_prefix: str = "./result/frame_handler_time"
):

    response_time_file_path = create_response_time_file(response_time_dir_path)
    Logger.info(f"响应时间文件保存路径：{response_time_file_path}")

    image_save_dir = create_image_save_dir(image_save_prefix)
    Logger.info(f"图片保存路径：{image_save_dir}")

    frame_rate_save_dir_main, frame_rate_save_dir_thread = create_frame_rate_save_dir(frame_rate_save_prefix)
    Logger.info(f"帧处理速率保存路径（无目标）：{frame_rate_save_dir_main}")
    Logger.info(f"帧处理速率保存路径（有目标）：{frame_rate_save_dir_thread}")

    image_time_file_path = create_image_time_save_dir(image_time_save_prefix)
    Logger.info(f"获取图片速率保存路径：{image_time_file_path}")

    frame_handler_time_file_path_main, frame_handler_time_file_path_thread = create_frame_handler_time_file(frame_handler_time_prefix)
    Logger.info(f"帧处理耗时保存路径（无目标）：{frame_handler_time_file_path_main}")
    Logger.info(f"帧处理耗时保存路径（有目标）：{frame_handler_time_file_path_thread}")

    time.sleep(sleep_time)

    # 创建多流管理器
    manager = MultiStreamManager()

    # 添加4个视频流，每个流可以设置不同的采样间隔
    manager.add_stream("stream9", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)
    manager.add_stream("stream10", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)
    manager.add_stream("stream11", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)
    manager.add_stream("stream12", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)

    # 创建线程池并保存为变量
    executor = ThreadPoolExecutor(max_workers=6)

    try:
        # 启动所有视频流
        manager.start_all_streams()

        frames = []

        frame_count = 0
        frame_start_time = time.time()

        frame_total_handler_time = 0.0

        while not manager_is_closed(manager):
            start_time = time.time()

            if frame_count >= 10:
                save_frame_rate(frame_count / (start_time - frame_start_time), frame_rate_save_dir_main)
                save_frame_handler_time(frame_total_handler_time / frame_count, frame_handler_time_file_path_main)
                frame_count = 0
                frame_total_handler_time = 0.0
                frame_start_time = start_time

            frame_count += 1

            # 清空帧列表
            frames.clear()
            futures = []
            # 遍历所有流，从每个流的队列中获取帧
            for stream_id in manager.simulators:
                # 提交任务到线程池
                future = executor.submit(fetch_frame_from_stream, stream_id, manager)
                futures.append(future)

            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    frames.append(result)

            # 拼接图片
            stitchedImage = stitch_images(frames)

            if stitchedImage is None:
                Logger.info("无视频流需要检测")
            else:
                # 调用推理模块，获取推理结果
                result = detect_image(stitchedImage.stitched_image, file_path=response_time_file_path)

                # 解析推理结果
                parsed_result = parse_inference_result(stitchedImage, result)

                if not parsed_result:
                    Logger.info("未解析到任何检测结果")
                else:
                    for inference_result in parsed_result:
                        # 如果stream_id不存在
                        if manager.has_stream(inference_result.stream_id) is False:
                            Logger.error(f"视频流{inference_result.stream_id}不存在")
                            continue

                        # 如果该视频流符号位为True，表示该视频流中存在目标，正由子线程处理，跳过
                        if manager.get_processing_flag(inference_result.stream_id):
                            continue

                        # 如果推理结果不为空，则处理该视频流
                        if inference_result.boxes.size > 0:
                            Logger.success(f"视频流{inference_result.stream_id}存在目标，开始单独处理")
                            # 设置该视频流的处理标志为True
                            manager.set_processing_flag(inference_result.stream_id, True)

                            simulator = manager.get_stream(inference_result.stream_id)

                            # 开启子线程处理该视频流
                            thread = threading.Thread(
                                target=processing_video_stream,
                                args=(
                                    manager,
                                    simulator,
                                    THREAD_SAMPLE_INTERVAL,
                                    KEYPOINTS_THRESH,
                                    response_time_file_path,
                                    image_save_dir,
                                    frame_rate_save_dir_thread,
                                    frame_handler_time_file_path_thread,
                                )
                            )
                            thread.daemon = True  # 设置为守护线程，主线程结束时该线程也会自动结束
                            thread.start()

            frame_total_handler_time += time.time() - start_time

            # 控制帧率
            control_loop_frequency(start_time, SAMPLE_INTERVAL)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 并优雅地停止所有流
        Logger.warning("接收到停止信号，正在停止所有视频流...")
        executor.shutdown()  # 显式关闭线程池
        manager.stop_all_streams()
        Logger.success("所有视频流已停止")

