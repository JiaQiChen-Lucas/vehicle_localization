import asyncio
import json
import threading
from datetime import datetime
from itertools import count

import numpy as np
import cv2
import requests
import time
import os
from io import BytesIO
import aiohttp
from typing import Dict, Any, Optional, Tuple
from .log import Logger
from .nacos_utils import get_nacos_config, get_services, Service

service_queue_length_list = []

class ServiceQueueLength:
    def __init__(self, host: str, port: str, device_type: str, healthy: bool, yolov8n_pose_len: int, yolov8s_seg_len: int, yolov8m_seg_len: int):
        self.host = host
        self.port = port
        self.device_type = device_type
        self.healthy = healthy
        self.yolov8n_pose_len = yolov8n_pose_len
        self.yolov8s_seg_len = yolov8s_seg_len
        self.yolov8m_seg_len = yolov8m_seg_len


async def get_queue_length_from_remote(host: str, port: str, device_type: str) -> Tuple[int, int, int]:
    url = f"http://{host}:{port}/queueLength"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=0.5) as response:
                if response.status == 200:
                    queue_info = await response.json()
                    return (
                        int(queue_info.get('yolov8n_pose_queue_length', 10)),
                        int(queue_info.get('yolov8s_seg_queue_length', 0)),
                        int(queue_info.get('yolov8m_seg_queue_length', 0))
                    )
                else:
                    raise Exception(f"HTTP {response.status}")
    except Exception as e:
        Logger.warning(f"请求 {url} 失败: {e}")
        if device_type == "OrangePi 5B":
            return 10, 10, 10
        else:
            return 10, 0, 0



async def fetch_and_update_queue_length(service):
    try:
        yolov8n_pose_len, yolov8s_seg_len, yolov8m_seg_len = await get_queue_length_from_remote(
            service.host, service.port, service.device_type
        )

        # 假设你有一个全局结构（如字典或列表）保存这些长度信息
        # 示例：更新对应 service_queue_length 的值
        found = False
        for sq in service_queue_length_list:
            if sq.host == service.host and sq.port == service.port and sq.device_type == service.device_type:
                sq.yolov8n_pose_len = yolov8n_pose_len
                sq.yolov8s_seg_len = yolov8s_seg_len
                sq.yolov8m_seg_len = yolov8m_seg_len
                sq.healthy = True
                # Logger.info(f"更新服务 {service.host}:{service.port} 的队列长度: {sq.yolov8n_pose_len, sq.yolov8s_seg_len, sq.yolov8m_seg_len}")
                found = True
                break
        if not found:
            service_queue_length_list.append(ServiceQueueLength(service.host, service.port, service.device_type, service.healthy, yolov8n_pose_len, yolov8s_seg_len, yolov8m_seg_len))
    except Exception as e:
        print(f"Error fetching queue length from {service.host}:{service.port} - {e}")


async def loop_update_service_queue_length(interval: float = 0.5) -> None:
    while True:
        start_time = time.time()
        services = get_services()

        for service_queue_length in service_queue_length_list:
            service_queue_length.healthy = False

        # 创建任务列表以并发执行
        tasks = []
        for service in services:
            if not service.healthy:
                continue

            # 异步发起远程调用
            task = asyncio.create_task(
                fetch_and_update_queue_length(service)
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        remaining_time = time.time() - start_time
        if remaining_time < interval:
            await asyncio.sleep(interval - remaining_time)

def start_background_updater_service_queue_length(interval: float = 0.5):
    def run_loop():
        asyncio.run(loop_update_service_queue_length(interval))

    updater_thread = threading.Thread(target=run_loop, daemon=True)
    updater_thread.start()

def getServiceQueueLengthList():
    return service_queue_length_list
