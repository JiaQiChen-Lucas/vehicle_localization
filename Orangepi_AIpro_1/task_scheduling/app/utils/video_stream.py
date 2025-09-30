import time
import threading
import cv2
from typing import Optional, Tuple
import numpy as np
from collections import deque
from .log import Logger
from .byte_track import BYTETracker

class VideoStreamSimulator:
    """模拟从单个源拉取视频流的类"""

    def __init__(self, stream_id: str, video_path: str, fps: int = 30, sample_interval: int = 10,
                 max_queue_size: int = 10, is_circular: bool = True, bytetrack_frame_rate: int = 30):
        """
        初始化视频流模拟器

        参数:
            stream_id: 视频流ID
            video_path: 本地视频文件路径
            fps: 模拟的帧率
            sample_interval: 采样间隔，每隔多少帧进行一次采样
            max_queue_size: 帧队列的最大长度，防止内存溢出
            is_circular: 是否循环读取视频
            bytetrack_frame_rate: 用于计算最大丢失帧数阈值，决定目标在被视为 “丢失” 前允许的最大未检测到帧数
        """
        self.stream_id = stream_id          # 视频流ID
        self.video_path = video_path        # 视频文件路径
        self.fps = fps                      # 模拟的帧率
        self.frame_delay = 1.0 / fps        # 每帧的延迟时间
        self.sample_interval = sample_interval  # 采样间隔
        self._stop_event = threading.Event()    # 用于控制线程停止的事件
        self._frame_queue = deque(maxlen=max_queue_size)  # 使用 deque 实现 LIFO 行为
        self._queue_lock = threading.Lock()     # 自定义锁，保护队列访问
        self.is_circular = is_circular
        self._thread = None         # 存储工作线程
        self._frame_count = 0       # 记录当前流的帧数
        self.bytetracker = BYTETracker(frame_rate=bytetrack_frame_rate)  # 目标跟踪器
        self.isDone = False

    def set_sample_interval(self, sample_interval: int):
        """
        设置新的采样间隔

        参数:
            sample_interval: 新的采样间隔值
        """
        self.sample_interval = sample_interval

    def start(self) -> None:
        """
        启动视频流

        如果视频流已经运行，则不会重复启动
        """
        if self._thread and self._thread.is_alive():
            Logger.warning(f"视频流 {self.stream_id} 已经在运行")
            return

        # 启动流线程
        self._thread = threading.Thread(target=self._stream_worker)
        self._thread.daemon = True
        self._thread.start()
        Logger.success(f"已启动视频流 {self.stream_id}")

    def stop(self) -> None:
        """
        停止视频流

        如果视频流未启动，则不执行任何操作
        """
        if self._stop_event.is_set():
            return  # 已经停止了

        self._stop_event.set()
        self.isDone = True

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        Logger.success(f"已停止视频流 {self.stream_id}")

    def get_first_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[int, np.ndarray]]:
        """
        获取最早的采样帧（FIFO）

        参数:
            timeout: 等待超时时间（秒），None表示无限等待

        返回:
            如果成功获取，返回 (帧号, 帧数据)，否则返回None
        """
        start_time = time.time()

        while True:
            with self._queue_lock:
                if self._frame_queue:
                    return self._frame_queue.popleft()  # 取最早入队的一帧

            # 如果没有帧且已超时，返回 None
            if timeout is not None and time.time() - start_time > timeout:
                return None

            # 避免 CPU 占用过高
            time.sleep(0.01)

    def get_latest_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[int, np.ndarray]]:
        """
        获取最新一帧（LIFO）

        参数:
            timeout: 等待超时时间（秒），None表示无限等待

        返回:
            如果成功获取，返回 (帧号, 帧数据)，否则返回None
        """
        start_time = time.time()
        while True:
            with self._queue_lock:
                if self._frame_queue:
                    return self._frame_queue.pop()  # 弹出最新帧

            # 如果没有帧且已超时，返回 None
            if timeout is not None and time.time() - start_time > timeout:
                return None

            # 避免 CPU 占用过高
            time.sleep(0.01)

    def clear_frame_queue(self) -> None:
        """
        清空帧队列中的所有内容
        """
        with self._queue_lock:
            self._frame_queue.clear()
        Logger.success(f"视频流 {self.stream_id} 的帧队列已清空")

    def get_queue_length(self) -> int:
        """
        获取当前帧队列的长度

        返回:
            帧队列的当前长度
        """
        with self._queue_lock:
            return len(self._frame_queue)

    def get_first_four_frames(self, timeout: Optional[float] = None) -> list:
        """
        获取队列中最前面的四个采样帧（FIFO）

        参数:
            timeout: 等待超时时间（秒），None表示无限等待

        返回:
            包含最多4个帧的列表，每个元素为 (帧号, 帧数据)，若超时则返回已获取的部分或空列表
        """
        start_time = time.time()
        frames = []

        while len(frames) < 4:
            with self._queue_lock:
                if self._frame_queue:
                    # 从队列头部取出一帧
                    frame = self._frame_queue.popleft()
                    frames.append(frame)
                else:
                    # 队列为空，跳出循环
                    break

            # 如果设置了超时且已超时，返回当前已获取的帧
            if timeout is not None and time.time() - start_time > timeout:
                break

            # 避免 CPU 占用过高
            time.sleep(0.01)

        return frames

    def _stream_worker(self) -> None:
        """
        视频流工作线程

        模拟持续拉取视频流并进行处理
        """
        Logger.info(f"视频流 {self.stream_id} 开始拉取，源文件: {self.video_path}\n")

        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            Logger.error(f"无法打开视频文件: {self.video_path}")
            return

        # 模拟视频流的持续拉取
        while not self._stop_event.is_set():
            start_time = time.time()

            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                if self.is_circular:
                    Logger.warning(f"视频流 {self.stream_id} 已到达文件末尾，重新开始播放")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                    ret, frame = cap.read()
                    if not ret:
                        Logger.error(f"视频流 {self.stream_id} 无法读取第一帧，退出")
                        break
                else:
                    Logger.success(f"视频流 {self.stream_id} 已播放完毕，停止拉取")
                    break

            # 更新帧计数器
            self._frame_count += 1
            frame_count = self._frame_count

            # 检查是否需要采样
            if frame_count % self.sample_interval == 0:
                self._put_frame_to_queue(frame_count, frame)

            # 控制帧率
            elapsed_time = time.time() - start_time
            if elapsed_time < self.frame_delay:
                time.sleep(self.frame_delay - elapsed_time)

        # 释放资源
        cap.release()
        self.isDone = True
        Logger.success(f"视频流 {self.stream_id} 拉取已停止")

    def _put_frame_to_queue(self, frame_number: int, frame: np.ndarray) -> None:
        """
        将采样帧放入队列供外部获取

        参数:
            frame_number: 帧号
            frame: 帧数据
        """
        try:
            with self._queue_lock:
                # 队列满时自动丢弃最旧的帧（由 maxlen 控制）
                self._frame_queue.append((frame_number, frame))
                # Logger.success(f"视频流 {self.stream_id} 已将第 {frame_number} 帧放入队列")
        except Exception as e:
            Logger.warning(f"视频流 {self.stream_id} 将帧放入队列时出错: {e}")



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

    def add_stream(self, stream_id: str, video_path: str, fps: int = 30, sample_interval: int = 10,
                   max_queue_size: int = 10, is_circular: bool = True, bytetrack_frame_rate: int = 30) -> None:
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
                                             sample_interval=sample_interval,
                                             max_queue_size=max_queue_size,
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