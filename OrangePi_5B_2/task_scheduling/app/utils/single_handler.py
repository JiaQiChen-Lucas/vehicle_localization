import json
from datetime import datetime
import os
import time
import threading
from collections import deque
# from .video_stream_online import MultiStreamManager, VideoStreamSimulator
from .video_stream_ffmpeg import MultiStreamManager, VideoStreamSimulator
from .log import Logger
from .detect import detect_image
from .processing import get_result, is_vehicle_static, save_image, count_keypoints_valid
from .position import check_position_is_out

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
        time.sleep(target_interval - elapsed)


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

def processing_video_stream(
        simulator: VideoStreamSimulator,
        KEYPOINTS_THRESH: float = 0.5,
        SAMPLE_INTERVAL: float = 1.0,
        THREAD_SAMPLE_INTERVAL: float = 0.5,
        RESPONSE_TIME_FILE_PATH: str = "./result/response_time/response_time.jsonl",
        IMAGE_SAVE_DIR_PATH: str = "./result/detect",
        FRAME_RATE_SAVE_DIR_PATH_MAIN: str = "./result/frame_rate_main.jsonl",
        FRAME_RATE_SAVE_DIR_PATH_THREAD: str = "./result/frame_rate_thread.jsonl",
        FRAME_HANDLER_TIME_FILE_PATH_MAIN: str = "./result/frame_handler_time_main.jsonl",
        FRAME_HANDLER_TIME_FILE_PATH_THREAD: str = "./result/frame_handler_time_thread.jsonl"
):
    """
    单独处理某个视频流
    :param simulator: 视频流操作器
    :param KEYPOINTS_THRESH: 关键点阈值
    :param RESPONSE_TIME_FILE_PATH: 响应时间存储文件位置
    :param IMAGE_SAVE_DIR_PATH: 图片存储目录
    :return:
    """
    if simulator is None:
        Logger.error("video_stream is None")
        return

    Logger.info(f"视频流 {simulator.stream_id} 开始处理")

    # 存储处理完的目标id
    processed_track_ids = set()

    frame_count_main = 0
    # 总时间（包含停顿时间）
    total_frame_time_main = 0.0
    # 总处理时间（不包含停顿时间）
    frame_total_handler_time_main = 0.0

    frame_count_thread = 0
    # 总时间（包含停顿时间）
    total_frame_time_thread = 0.0
    # 总处理时间（不包含停顿时间）
    frame_total_handler_time_thread = 0.0

    while not simulator.isDone:
        start_time = time.time()

        if frame_count_main >= 10:
            save_frame_rate(frame_count_main / total_frame_time_main, FRAME_RATE_SAVE_DIR_PATH_MAIN)
            save_frame_handler_time(frame_total_handler_time_main / frame_count_main, FRAME_HANDLER_TIME_FILE_PATH_MAIN)
            frame_count_main = 0
            frame_total_handler_time_main = 0.0
            total_frame_time_main = 0.0

        current_frame = simulator.get_frame()

        if current_frame is not None:
            # 调用推理模块，获取推理结果
            result = detect_image(current_frame, file_path=RESPONSE_TIME_FILE_PATH)

            # 解析推理结果
            target_info = get_result(current_frame, result, simulator)

            # 如果没有推理结果或推理结果中没有目标，则跳过当前帧
            if target_info is None or target_info.boxes.shape[0] == 0:
                # Logger.info("未检测到目标")
                frame_total_handler_time_main += time.time() - start_time
                # 控制帧率
                control_loop_frequency(start_time, SAMPLE_INTERVAL)
                frame_count_main += 1
                total_frame_time_main += time.time() - start_time
                continue

            # 创建一个最大容量为4的队列
            image_result_queue = deque(maxlen=4)
            # 添加第一个结果
            image_result_queue.append(target_info)

            # 连续几帧没有检测到目标
            object_miss_count = 0

            Logger.info(f"视频流 {simulator.stream_id} 检测到目标，开始跟踪")

            frame_total_handler_time_main += time.time() - start_time
            control_loop_frequency(start_time, SAMPLE_INTERVAL)
            frame_count_main += 1
            total_frame_time_main += time.time() - start_time

            # 目标检测与跟踪
            while not simulator.isDone:
                start_time_2 = time.time()
                if frame_count_thread >= 10:
                    save_frame_rate(frame_count_thread / total_frame_time_thread, FRAME_RATE_SAVE_DIR_PATH_THREAD)
                    save_frame_handler_time(frame_total_handler_time_thread / frame_count_thread, FRAME_HANDLER_TIME_FILE_PATH_THREAD)
                    frame_count_thread = 0
                    frame_total_handler_time_thread = 0.0
                    total_frame_time_thread = 0.0

                # 读取一帧
                current_frame = simulator.get_frame()

                if current_frame is None:
                    Logger.error(f"视频流 {simulator.stream_id} 未读取到任何帧")
                    # 控制速率
                    frame_total_handler_time_thread += time.time() - start_time_2
                    control_loop_frequency(start_time_2, THREAD_SAMPLE_INTERVAL)
                    frame_count_thread += 1
                    total_frame_time_thread += time.time() - start_time_2
                    continue

                # 推理
                result = detect_image(current_frame, file_path=RESPONSE_TIME_FILE_PATH)

                # 解析推理结果
                target_info = get_result(current_frame, result, simulator)

                # 如果没有检测到目标
                if target_info is None or target_info.boxes.shape[0] == 0:
                    Logger.info(f"视频流 {simulator.stream_id} 未检测到目标")
                    object_miss_count += 1
                    if object_miss_count >= 4:
                        Logger.info(f"视频流 {simulator.stream_id} 连续{object_miss_count}帧未解析到任何检测结果")
                        break
                    frame_total_handler_time_thread += time.time() - start_time_2
                    # 控制速率
                    control_loop_frequency(start_time_2, THREAD_SAMPLE_INTERVAL)
                    frame_count_thread += 1
                    total_frame_time_thread += time.time() - start_time_2
                    continue

                # 重置
                object_miss_count = 0

                # 添加到结果队列
                image_result_queue.append(target_info)

                # 判断目标是否静止（可能一张图片内有多个目标，只关注检测框最大的那个）
                if is_vehicle_static(image_result_queue):
                    # 清空队列
                    image_result_queue.clear()

                    Logger.info(f"视频流 {simulator.stream_id} 目标已静止")

                    frame_total_handler_time_thread += time.time() - start_time_2
                    control_loop_frequency(start_time_2, THREAD_SAMPLE_INTERVAL)
                    frame_count_thread += 1
                    total_frame_time_thread += time.time() - start_time_2

                    max_detect_count = 3

                    # 最多读取 max_detect_count 检验
                    while max_detect_count > 0:
                        start_time_3 = time.time()
                        max_detect_count -= 1
                        # 读取一帧
                        current_frame = simulator.get_frame()

                        if current_frame is None:
                            Logger.error(f"视频流 {simulator.stream_id} 未读取到任何帧")
                            frame_total_handler_time_thread += time.time() - start_time_3
                            # 控制速率
                            control_loop_frequency(start_time_3, THREAD_SAMPLE_INTERVAL)
                            frame_count_thread += 1
                            total_frame_time_thread += time.time() - start_time_3
                            continue

                        # 推理
                        result = detect_image(current_frame, file_path=RESPONSE_TIME_FILE_PATH)

                        # 解析推理结果
                        target_info = get_result(current_frame, result, simulator)

                        if target_info is None or target_info.boxes.shape[0] == 0:
                            Logger.info(f"视频流 {simulator.stream_id} 未检测到目标")
                            frame_total_handler_time_thread += time.time() - start_time_3
                            # 控制速率
                            control_loop_frequency(start_time_3, THREAD_SAMPLE_INTERVAL)
                            frame_count_thread += 1
                            total_frame_time_thread += time.time() - start_time_3
                            continue

                        # 如果track_id已经被检测过
                        if target_info.track_ids[0] in processed_track_ids:
                            Logger.info(f"视频流 {simulator.stream_id} 目标已处理过")
                            frame_total_handler_time_thread += time.time() - start_time_3
                            control_loop_frequency(start_time_3, THREAD_SAMPLE_INTERVAL)
                            frame_count_thread += 1
                            total_frame_time_thread += time.time() - start_time_3
                            break

                        # 统计有效关键点数量
                        valid_counts = count_keypoints_valid(target_info.keypoints, KEYPOINTS_THRESH)

                        if valid_counts[0] < 2:
                            Logger.info(f"视频流 {simulator.stream_id} 目标关键点数量不足")
                            frame_total_handler_time_thread += time.time() - start_time_3
                            # 控制速率
                            control_loop_frequency(start_time_3, THREAD_SAMPLE_INTERVAL)
                            frame_count_thread += 1
                            total_frame_time_thread += time.time() - start_time_3
                            continue

                        # 判断是否越界
                        if check_position_is_out(image_result=target_info, stream_id=simulator.stream_id,
                                                 RESULT_DIR=IMAGE_SAVE_DIR_PATH, is_save=True,
                                                 KEYPOINTS_THRESH=KEYPOINTS_THRESH):
                            Logger.info(f"视频流 {simulator.stream_id} 目标越界")
                            frame_total_handler_time_thread += time.time() - start_time_3
                            # 控制帧率
                            control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                            frame_count_thread += 1
                            total_frame_time_thread += time.time() - start_time_3
                            continue

                        # 绘制并保存图片
                        save_image(
                            parsed_result=[target_info],
                            stream_id=simulator.stream_id,
                            KEYPOINTS_THRESH=KEYPOINTS_THRESH,
                            RESULT_DIR=f"{IMAGE_SAVE_DIR_PATH}/success",
                            IS_LOGGING=True
                        )

                        # 加入processed_track_ids
                        processed_track_ids.add(target_info.track_ids[0])
                        Logger.success(f"{simulator.stream_id} 目标已处理完毕，track_id: {target_info.track_ids}, box: {target_info.boxes}, cls: {target_info.classes}, score: {target_info.scores}, keypoints: {target_info.keypoints}")

                        # 控制帧率
                        frame_total_handler_time_thread += time.time() - start_time_3
                        control_loop_frequency(start_time, THREAD_SAMPLE_INTERVAL)
                        frame_count_thread += 1
                        total_frame_time_thread += time.time() - start_time_3

                        # 退出内循环
                        break
                else:
                    frame_total_handler_time_thread += time.time() - start_time_2
                    control_loop_frequency(start_time_2, THREAD_SAMPLE_INTERVAL)
                    frame_count_thread += 1
                    total_frame_time_thread += time.time() - start_time_2
        else:
            # 控制帧率
            control_loop_frequency(start_time, SAMPLE_INTERVAL)

    Logger.success(f"视频流 {simulator.stream_id} 处理完毕")


def single_handler_start(
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
    manager.add_stream("stream5", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)
    manager.add_stream("stream6", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)
    manager.add_stream("stream7", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)
    manager.add_stream("stream8", is_circular=True, bytetrack_frame_rate=5, image_time_save_path=image_time_file_path)

    try:
        # 启动所有视频流
        manager.start_all_streams()

        for stream_id in manager.simulators:
            simulator = manager.get_stream(stream_id)
            if not simulator:
                continue

            # 开启子线程处理该视频流
            thread = threading.Thread(
                target=processing_video_stream,
                args=(simulator, KEYPOINTS_THRESH, SAMPLE_INTERVAL, THREAD_SAMPLE_INTERVAL, response_time_file_path, image_save_dir, frame_rate_save_dir_main, frame_rate_save_dir_thread, frame_handler_time_file_path_main, frame_handler_time_file_path_thread, )
            )
            thread.daemon = True  # 设置为守护线程，主线程结束时该线程也会自动结束
            thread.start()

        # 主线程保持运行，直到收到中断信号
        while not manager_is_closed(manager):
            time.sleep(1)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 并优雅地停止所有流
        Logger.warning("接收到停止信号，正在停止所有视频流...")
        manager.stop_all_streams()
        Logger.success("所有视频流已停止")
