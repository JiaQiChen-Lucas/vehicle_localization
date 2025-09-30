import time
import threading
import cv2
from typing import Optional, Tuple

import numpy as np

from .log import Logger
from .byte_track import BYTETracker

cv2.setNumThreads(1)    # 限制OpenCV为单线程
cv2.ocl.setUseOpenCL(False)     # 禁用OpenCL加速

class VideoStreamSimulator:
    """模拟从单个源拉取视频流的类"""

    def __init__(self, stream_id: str, video_path: str, fps: int = 30, is_circular: bool = True, bytetrack_frame_rate: int = 30):
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
        self.video_path = video_path        # 视频文件路径
        self.fps = fps                      # 模拟的帧率
        self.is_circular = is_circular
        self.start_time = None
        self.cap = None
        self.bytetracker = BYTETracker(frame_rate=bytetrack_frame_rate)  # 目标跟踪器
        self.isStarted = False
        self.isDone = False
        self.total_frames = -1     # 视频总帧数

    def start(self) -> None:
        """
        启动视频流

        如果视频流已经运行，则不会重复启动
        """
        if self.isStarted:
            Logger.warning(f"视频流 {self.stream_id} 已经启动，请勿重复启动")
            return

        if self.video_path is None or self.video_path == "":
            raise ValueError(f"视频流 {self.stream_id} 的视频路径不能为空")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"视频流 {self.stream_id} 无法打开视频文件: {self.video_path}")

        self.isStarted = True

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        Logger.info(f"视频流 {self.stream_id} 的总帧数: {self.total_frames}")

        self.start_time = time.time()
        Logger.success(f"已启动视频流 {self.stream_id}")

    def stop(self) -> None:
        """
        停止视频流

        如果视频流未启动，则不执行任何操作
        """
        if not self.isStarted or self.isDone:
            return  # 已经停止了

        self.isDone = True
        self.cap.release()  # 释放资源
        Logger.success(f"已停止视频流 {self.stream_id}")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        根据当前时间戳和帧率跳转到对应的视频帧，返回 (frame_number, frame)

        返回:
            如果成功获取帧，则返回 (帧号, 帧数据)，否则返回 None
        """
        start_time = time.time()

        if not self.isStarted or self.isDone:
            Logger.error(f"视频流 {self.stream_id} 未启动或已停止")
            return None

        if self.total_frames == -1:
            raise ValueError(f"视频流 {self.stream_id} 未初始化")

        # 当前时间与开始时间差值（秒）
        elapsed_seconds = time.time() - self.start_time

        # 当前读到多少帧
        current_frames = int(self.fps * elapsed_seconds)

        if self.is_circular:
            # 如果是循环播放，对视频总帧数取余
            current_frames = current_frames % self.total_frames

        if current_frames > self.total_frames:
            Logger.warning(f"视频流 {self.stream_id} 已播放完毕，无法再读取帧")
            self.stop()
            return None

        # 设置帧位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frames)

        # 读取帧
        ret, current_frame = self.cap.read()
        if not ret:
            Logger.warning(f"视频流 {self.stream_id} 无法读取第 {current_frames} 帧")
            return None

        Logger.success(f"视频流 {self.stream_id} 读取第 {current_frames} 帧成功")
        Logger.info(f"视频流 {self.stream_id} 读取一帧耗时：{time.time() - start_time}")

        return current_frame


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

    def add_stream(self, stream_id: str, video_path: str, fps: int = 30, is_circular: bool = True, bytetrack_frame_rate: int = 30) -> None:
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
                                             video_path=video_path,
                                             fps=fps,
                                             is_circular=is_circular,
                                             bytetrack_frame_rate=bytetrack_frame_rate)

            self.simulators[stream_id] = StreamInfo(simulator, False)
            Logger.success(f"已添加视频流 {stream_id}")

    def start_all_streams(self) -> None:
        """启动所有视频流"""
        with self._lock:
            for stream_id, info in self.simulators.items():
                info.simulator.start()

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
    manager.add_stream("stream1", "../video/video_5m_1.mp4", is_circular=False, bytetrack_frame_rate=5)
    manager.add_stream("stream2", "../video/video_5m_2.mp4", is_circular=False, bytetrack_frame_rate=5)
    manager.add_stream("stream3", "../video/video_5m_3.mp4", is_circular=False, bytetrack_frame_rate=5)
    manager.add_stream("stream4", "../video/video_5m_4.mp4", is_circular=False, bytetrack_frame_rate=5)

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

