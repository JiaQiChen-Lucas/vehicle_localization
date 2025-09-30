import os
import time
import threading
import cv2
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from queue import Queue, Empty


# class VideoStreamSimulator:
#     """模拟从单个源拉取视频流的类"""
#
#     def __init__(self, stream_id: str, video_path: str, fps: int = 30, sample_interval: int = 10,
#                  max_queue_size: int = 10, is_circular: bool = True):
#         """
#         初始化视频流模拟器
#
#         参数:
#             stream_id: 视频流ID
#             video_path: 本地视频文件路径
#             fps: 模拟的帧率
#             sample_interval: 采样间隔，每隔多少帧进行一次采样
#             max_queue_size: 帧队列的最大长度，防止内存溢出
#             is_loop: 是否循环读取视频
#         """
#         self.stream_id = stream_id          # 视频流ID
#         self.video_path = video_path        # 视频文件路径
#         self.fps = fps                      # 模拟的帧率
#         self.frame_delay = 1.0 / fps        # 每帧的延迟时间
#         self.sample_interval = sample_interval  # 采样间隔
#         self._stop_event = threading.Event()    # 用于控制线程停止的事件
#         self._frame_queue = Queue(maxsize=max_queue_size)  # 用于存储采样帧的队列
#         self.is_circular = is_circular
#         self._thread = None         # 存储工作线程
#         self._frame_count = 0  # 记录当前流的帧数
#
#     def set_sample_interval(self, sample_interval: int):
#         """
#         设置新的采样间隔
#
#         参数:
#             sample_interval: 新的采样间隔值
#         """
#         # 会有线程安全问题，但是不是那么重要
#         self.sample_interval = sample_interval
#
#     def start(self) -> None:
#         """
#         启动视频流
#
#         如果视频流已经运行，则不会重复启动
#         """
#         if self._thread and self._thread.is_alive():
#             print(f"视频流 {self.stream_id} 已经在运行")
#             return
#
#         # 启动流线程
#         self._thread = threading.Thread(target=self._stream_worker)
#         self._thread.daemon = True
#         self._thread.start()
#         print(f"已启动视频流 {self.stream_id}")
#
#     def stop(self) -> None:
#         """
#         停止视频流
#
#         如果视频流未启动，则不执行任何操作
#         """
#         if self._stop_event.is_set():
#             return  # 已经停止了
#
#         self._stop_event.set()
#
#         if self._thread and self._thread.is_alive():
#             self._thread.join(timeout=2.0)
#
#         print(f"已停止视频流 {self.stream_id}")
#
#     def get_next_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[int, np.ndarray]]:
#         """
#         获取下一个采样帧
#
#         参数:
#             timeout: 等待超时时间（秒），None表示无限等待
#
#         返回:
#             如果成功获取，返回 (帧号, 帧数据)，否则返回None
#         """
#         try:
#             # Queue.get() 是线程安全的
#             return self._frame_queue.get(block=True, timeout=timeout)
#         except Empty:
#             return None
#
#     def get_latest_frame(self) -> Optional[Tuple[int, np.ndarray]]:
#         """
#         获取队列中最新的一个帧
#
#         返回:
#             如果队列非空，返回 (帧号, 帧数据)，否则返回 None
#         """
#         with self._frame_queue.mutex:  # Queue 内部锁，保证线程安全
#             if not self._frame_queue.queue:
#                 return None
#             return self._frame_queue.queue[-1]  # 获取最后一个元素
#
#     def clear_frame_queue(self) -> None:
#         """
#         清空帧队列中的所有内容
#         """
#         with self._frame_queue.mutex:   # Queue 内部锁，保证线程安全
#             self._frame_queue.queue.clear()
#         print(f"视频流 {self.stream_id} 的帧队列已清空")
#
#
#     def _stream_worker(self) -> None:
#         """
#         视频流工作线程
#
#         模拟持续拉取视频流并进行处理
#         """
#         print(f"视频流 {self.stream_id} 开始拉取，源文件: {self.video_path}")
#
#         # 打开视频文件
#         cap = cv2.VideoCapture(self.video_path)
#         if not cap.isOpened():
#             print(f"无法打开视频文件: {self.video_path}")
#             return
#
#         # 模拟视频流的持续拉取
#         while not self._stop_event.is_set():
#             start_time = time.time()
#
#             # 读取视频帧
#             ret, frame = cap.read()
#             if not ret:
#                 if self.is_circular:
#                     print(f"视频流 {self.stream_id} 已到达文件末尾，重新开始播放")
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
#                     ret, frame = cap.read()
#                     if not ret:
#                         print(f"视频流 {self.stream_id} 无法读取第一帧，退出")
#                         break
#                 else:
#                     print(f"视频流 {self.stream_id} 已播放完毕，停止拉取")
#                     break
#
#             # 更新帧计数器
#             self._frame_count += 1
#             frame_count = self._frame_count
#
#             # 检查是否需要采样
#             if frame_count % self.sample_interval == 0:
#                 self._put_frame_to_queue(frame_count, frame)
#
#             # 控制帧率
#             elapsed_time = time.time() - start_time
#             if elapsed_time < self.frame_delay:
#                 time.sleep(self.frame_delay - elapsed_time)
#
#         # 释放资源
#         cap.release()
#         print(f"视频流 {self.stream_id} 拉取已停止")
#
#     def _put_frame_to_queue(self, frame_number: int, frame: np.ndarray) -> None:
#         """
#         将采样帧放入队列供外部获取
#
#         参数:
#             frame_number: 帧号
#             frame: 帧数据
#         """
#         try:
#             # Python 的 queue.Queue 类本身是为多线程环境设计的，其 get_nowait() 和 put_nowait() 方法都是原子操作。
#             # 如果队列已满，丢弃最旧的帧
#             if self._frame_queue.full():
#                 self._frame_queue.get_nowait()
#
#             self._frame_queue.put_nowait((frame_number, frame))
#             print(f"视频流 {self.stream_id} 已将第 {frame_number} 帧放入队列")
#         except Exception as e:
#             print(f"视频流 {self.stream_id} 将帧放入队列时出错: {e}")


class VideoStreamSimulator:
    """模拟从单个源拉取视频流的类"""

    def __init__(self, stream_id: str, video_path: str, fps: int = 30, sample_interval: int = 10,
                 max_queue_size: int = 10, is_circular: bool = True):
        """
        初始化视频流模拟器

        参数:
            stream_id: 视频流ID
            video_path: 本地视频文件路径
            fps: 模拟的帧率
            sample_interval: 采样间隔，每隔多少帧进行一次采样
            max_queue_size: 帧队列的最大长度，防止内存溢出
            is_loop: 是否循环读取视频
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
            print(f"视频流 {self.stream_id} 已经在运行")
            return

        # 启动流线程
        self._thread = threading.Thread(target=self._stream_worker)
        self._thread.daemon = True
        self._thread.start()
        print(f"已启动视频流 {self.stream_id}")

    def stop(self) -> None:
        """
        停止视频流

        如果视频流未启动，则不执行任何操作
        """
        if self._stop_event.is_set():
            return  # 已经停止了

        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        print(f"已停止视频流 {self.stream_id}")

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
        print(f"视频流 {self.stream_id} 的帧队列已清空")

    def _stream_worker(self) -> None:
        """
        视频流工作线程

        模拟持续拉取视频流并进行处理
        """
        print(f"视频流 {self.stream_id} 开始拉取，源文件: {self.video_path}")

        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {self.video_path}")
            return

        # 模拟视频流的持续拉取
        while not self._stop_event.is_set():
            start_time = time.time()

            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                if self.is_circular:
                    print(f"视频流 {self.stream_id} 已到达文件末尾，重新开始播放")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                    ret, frame = cap.read()
                    if not ret:
                        print(f"视频流 {self.stream_id} 无法读取第一帧，退出")
                        break
                else:
                    print(f"视频流 {self.stream_id} 已播放完毕，停止拉取")
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
        print(f"视频流 {self.stream_id} 拉取已停止")

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
                print(f"视频流 {self.stream_id} 已将第 {frame_number} 帧放入队列")
        except Exception as e:
            print(f"视频流 {self.stream_id} 将帧放入队列时出错: {e}")



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
                   max_queue_size: int = 10, is_circular: bool = True) -> None:
        """
        添加视频流

        参数:
            stream_id: 视频流ID
            video_path: 本地视频文件路径
            fps: 模拟的帧率
            sample_interval: 采样间隔，每隔多少帧进行一次采样
            max_queue_size: 帧队列的最大长度
        """
        with self._lock:
            if stream_id in self.simulators:
                print(f"视频流 {stream_id} 已存在")
                return

            simulator = VideoStreamSimulator(stream_id, video_path, fps, sample_interval, max_queue_size, is_circular)
            self.simulators[stream_id] = StreamInfo(simulator, False)
            print(f"已添加视频流 {stream_id}")

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
                print(f"视频流 {stream_id} 标志位已更新为 {contain_object}")
            else:
                print(f"视频流 {stream_id} 不存在，无法更新标志位")

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
                print(f"视频流 {stream_id} 不存在，无法获取标志位")
                return False
            return info.contain_object


def external_processing_example(manager: MultiStreamManager) -> None:
    """
    示例：如何在外部代码中获取和处理多个视频流的帧

    参数:
        manager: 多流管理器实例
    """
    print("\n=== 外部处理示例 ===")

    try:
        while True:
            # 遍历所有流，从每个流的队列中获取帧
            for stream_id in manager.simulators:
                # print(f"正在处理视频流 {stream_id}")
                simulator = manager.get_stream(stream_id)
                if not simulator:
                    continue

                # 获取帧，超时时间设为0.1秒
                result = simulator.get_first_frame(timeout=0.1)

                if result is not None:
                    frame_number, frame = result
                    print(f"从视频流 {stream_id} 获取到第 {frame_number} 帧")

                    # 在这里进行实际的处理
                    print(f"处理视频流 {stream_id}: 帧形状 {frame.shape}")

                    # 示例：保存帧为图像
                    # cv2.imwrite(f"frame_{stream_id}_{frame_number}.jpg", frame)
                    # print(f"  已保存为 'frame_{stream_id}_{frame_number}.jpg'")
                else:
                    # 没有可用帧
                    print(f"视频流 {stream_id} 当前没有可用帧")

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n外部处理已停止")


def main():
    """主函数，演示如何使用多视频流模拟器"""
    # 创建多流管理器
    manager = MultiStreamManager()

    # 添加4个视频流，每个流可以设置不同的采样间隔
    manager.add_stream("stream1", "./video/truck_1.mp4", sample_interval=10, is_circular=True)
    manager.add_stream("stream2", "./video/truck_2.mp4", sample_interval=15, is_circular=True)
    manager.add_stream("stream3", "./video/truck_1.mp4", sample_interval=20, is_circular=True)
    manager.add_stream("stream4", "./video/truck_2.mp4", sample_interval=25, is_circular=True)

    try:
        # 启动所有视频流
        manager.start_all_streams()

        # 启动外部处理线程
        processing_thread = threading.Thread(target=external_processing_example, args=(manager,))
        processing_thread.daemon = True
        processing_thread.start()

        # 主程序保持运行
        print("正在模拟拉取多个视频流，按 Ctrl+C 停止...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 并优雅地停止所有流
        print("\n接收到停止信号，正在停止所有视频流...")
        manager.stop_all_streams()
        print("所有视频流已停止")


if __name__ == "__main__":
    main()