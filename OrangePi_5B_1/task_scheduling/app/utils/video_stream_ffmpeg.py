import json
import os
import time
import threading
from datetime import datetime

import cv2
from typing import Optional, Tuple

import numpy as np

from .log import Logger
from .byte_track import BYTETracker
import requests

REMOTE_HOST = "192.168.3.101:8000"
FFMPEG_PATH = "rtsp://192.168.3.101:8554/video"

class VideoStreamSimulator:
    """模拟从单个源拉取视频流的类"""

    def __init__(
            self,
            stream_id: str,
            is_circular: bool = True,
            bytetrack_frame_rate: int = 30,
            image_time_save_path: str = "./result/image_rate.jsonl",
    ):
        """
        初始化视频流模拟器

        参数:
            stream_id: 视频流ID
            video_path: 本地视频文件路径
            fps: 模拟的帧率
            is_circular: 是否循环读取视频
            bytetrack_frame_rate: 用于计算最大丢失帧数阈值，决定目标在被视为 “丢失” 前允许的最大未检测到帧数
        """
        self.stream_id = stream_id          # 视频流ID
        self.is_circular = is_circular
        self.image_time_save_path = image_time_save_path
        os.makedirs(os.path.dirname(image_time_save_path), exist_ok=True)
        self._stop_event = threading.Event()
        self.thread = None
        self.bytetracker = BYTETracker(frame_rate=bytetrack_frame_rate)  # 目标跟踪器
        self.isStarted = False
        self.isDone = False
        self.cap = None
        self.current_frame = None

    def _stream_worker(self) -> None:
        url_path = f"{FFMPEG_PATH}/{self.stream_id}"
        self.cap = cv2.VideoCapture(url_path, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            Logger.success(f"已开始拉取视频流：{url_path}")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            Logger.error(f"无法拉取视频流：{url_path}")
            self.isStarted = False
            return
        self.isStarted = True

        frame_count = 0

        retry_count = 0
        max_retries = 3

        start_time = time.time()

        try:
            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    Logger.warning(f"视频流 {self.stream_id} 解码失败")
                    self.current_frame = None
                    retry_count += 1
                    if retry_count >= max_retries:
                        retry_count = 0
                        Logger.error(f"视频流 {self.stream_id} 达到最大重试次数，尝试重新连接...")
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(url_path, cv2.CAP_FFMPEG)
                        if self.cap.isOpened():
                            Logger.success(f"已开始拉取视频流：{url_path}")
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                        else:
                            Logger.error(f"无法拉取视频流：{url_path}")
                    continue
                self.current_frame = frame

                frame_count += 1
                if frame_count % 30 == 0:
                    # 保存图片时间
                    log_data = {
                        "timestamp": datetime.now().isoformat(),
                        "fps_for_30_frames": 30 / (time.time() - start_time)  # 平均帧率
                    }

                    with open(self.image_time_save_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

                    frame_count = 0
                    start_time = time.time()


        except Exception as e:
            Logger.error(f"视频流 {self.stream_id} 后台读取异常：{e}")
        finally:
            self.cap.release()
            self.isStarted = False
            Logger.info(f"视频流 {self.stream_id} 已释放")


    def start(self) -> None:
        """
        启动视频流（通过远程服务器控制）

        如果视频流已经运行，则不会重复启动
        """
        if self.isStarted:
            Logger.warning(f"视频流 {self.stream_id} 已经启动，请勿重复启动")
            return

        # 给REMOTE_HOST发请求
        # 构造请求数据
        start_url = f"http://{REMOTE_HOST}/startStream"
        payload = {
            "stream_id": self.stream_id,
            "is_circular": self.is_circular
        }

        try:
            response = requests.get(start_url, params=payload, timeout=5)
            if response.status_code == 200 and response.json().get("success"):
                time.sleep(5)  # 等待流准备就绪
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._stream_worker, daemon=True)
                self._thread.start()
                Logger.success(f"视频流 {self.stream_id} 后台线程已启动")
            else:
                error_msg = response.json().get("message", "未知错误")
                raise RuntimeError(f"启动视频流失败：{error_msg}")
        except requests.exceptions.RequestException as e:
            Logger.error(f"与远程服务通信失败：{e}")
            raise

    def stop(self) -> None:
        if not self.isStarted or self.isDone:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)  # 最多等待3秒

        stop_url = f"http://{REMOTE_HOST}/stopStream"
        payload = {"stream_id": self.stream_id}

        try:
            response = requests.get(stop_url, params=payload, timeout=5)
            if response.status_code == 200 and response.json().get("success"):
                self.isDone = True
                Logger.success(f"已停止视频流 {self.stream_id}")
            else:
                Logger.error(f"停止视频流失败：{response.json().get('message', '未知错误')}")
        except requests.exceptions.RequestException as e:
            Logger.error(f"停止视频流失败：{e}")

        self.isStarted = False

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.isStarted or self.isDone:
            Logger.error(f"视频流 {self.stream_id} 未启动或已停止")
            return None

        return self.current_frame


class StreamInfo:
    def __init__(self, simulator: VideoStreamSimulator, contain_object: bool):
        self.simulator = simulator
        self.contain_object = contain_object


class MultiStreamManager:
    """管理多个视频流模拟器的类"""

    def __init__(self):
        """初始化多流管理器"""
        self.simulators = {}
        self._lock = threading.Lock()

    def add_stream(
            self,
            stream_id: str,
            is_circular: bool = True,
            bytetrack_frame_rate: int = 30,
            image_time_save_path: str = "./result/image_rate.jsonl"
    ) -> None:
        """
        添加视频流

        参数:
            stream_id: 视频流ID
            video_path: 本地视频文件路径
            fps: 模拟的帧率
            sample_interval: 采样间隔，每隔多少帧进行一次采样
            max_queue_size: 帧队列的最大长度
            is_circular: 是否循环读取视频
            bytetrack_frame_rate: 用于计算最大丢失帧数阈值，决定目标在被视为 “丢失” 前允许的最大未检测到帧数
        """
        with self._lock:
            if stream_id in self.simulators:
                Logger.warning(f"视频流 {stream_id} 已存在")
                return

            simulator = VideoStreamSimulator(stream_id=stream_id,
                                             is_circular=is_circular,
                                             bytetrack_frame_rate=bytetrack_frame_rate,
                                             image_time_save_path=image_time_save_path)

            self.simulators[stream_id] = StreamInfo(simulator, False)
            Logger.success(f"已添加视频流 {stream_id}")

    def start_all_streams(self) -> None:
        """启动所有视频流"""
        with self._lock:
            for stream_id, info in self.simulators.items():
                info.simulator.start()
        time.sleep(2)

    def stop_all_streams(self) -> None:
        """停止所有视频流"""
        with self._lock:
            for stream_id, info in self.simulators.items():
                info.simulator.stop()

    def get_stream(self, stream_id: str) -> Optional[VideoStreamSimulator]:
        """
        获取指定ID的视频流模拟器

        参数:
            stream_id: 视频流ID

        返回:
            视频流模拟器实例，如果不存在则返回None
        """
        with self._lock:
            info = self.simulators.get(stream_id)
            return info.simulator if info else None

    def set_processing_flag(self, stream_id: str, contain_object: bool) -> None:
        """
        设置视频流是否存在目标

        参数:
            stream_id: 视频流ID
            contain_object: 是否包含物体
        """
        with self._lock:
            if stream_id in self.simulators:
                self.simulators[stream_id].contain_object = contain_object
                Logger.success(f"视频流 {stream_id} 标志位已更新为 {contain_object}")
            else:
                Logger.error(f"视频流 {stream_id} 不存在，无法更新标志位")

    def get_processing_flag(self, stream_id: str) -> bool:
        """
        获取视频流是否存在目标

        参数:
            stream_id: 视频流ID

        返回:
            是否包含物体
        """
        with self._lock:
            info = self.simulators.get(stream_id)
            if info is None:
                Logger.error(f"视频流 {stream_id} 不存在，无法获取标志位")
                return False
            return info.contain_object

    def has_stream(self, stream_id: str) -> bool:
        """
        判断是否存在指定的视频流

        参数:
            stream_id: 视频流ID

        返回:
            bool: 如果存在该视频流，返回 True；否则返回 False
        """
        with self._lock:
            return stream_id in self.simulators


if __name__ == "__main__":
    # 创建多流管理器
    manager = MultiStreamManager()

    # 添加4个视频流，每个流可以设置不同的采样间隔
    manager.add_stream("stream1", is_circular=False, bytetrack_frame_rate=5)
    manager.add_stream("stream2", is_circular=False, bytetrack_frame_rate=5)
    manager.add_stream("stream3", is_circular=False, bytetrack_frame_rate=5)
    manager.add_stream("stream4", is_circular=False, bytetrack_frame_rate=5)

    try:
        # 启动所有视频流
        manager.start_all_streams()

        time.sleep(20)

        frame = manager.get_stream("stream1").get_frame()

        if frame is not None:
            print("Received frame from stream1")

            print(frame)

            cv2.imwrite("/home/orangepi/rockchip/rknn_toolkit_lite2/examples/task_scheduling/app/result/task_scheduling_test.jpg", frame)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 并优雅地停止所有流
        Logger.warning("接收到停止信号，正在停止所有视频流...")
        manager.stop_all_streams()
        Logger.success("所有视频流已停止")

