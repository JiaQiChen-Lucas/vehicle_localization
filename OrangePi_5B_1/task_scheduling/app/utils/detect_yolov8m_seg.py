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
from typing import Dict, Any, Optional, Tuple
from .log import Logger
from .nacos_utils import get_nacos_config, get_services, Service
from .get_service_queue_length import getServiceQueueLengthList

_index_gen = count()
_choose_lock = threading.Lock()

def yolov8m_seg_choose_service_strategy_by_circle() -> Optional[Service]:
    """
    使用线程安全的 itertools.count 实现轮询索引生成
    """
    services = get_services()
    if not services:
        Logger.error("没有可用的推理服务")
        return None

    healthy_services = [s for s in services if s.healthy and s.host and s.port and s.device_type == "OrangePi 5B"]
    if not healthy_services:
        Logger.warning("没有健康的推理服务 或 不满足设备类型条件")
        return None

    with _choose_lock:
        index = next(_index_gen) % len(healthy_services)

    selected = healthy_services[index]
    Logger.info(f"yolov8m-seg 使用轮询策略选择了服务: {selected.host}:{selected.port}")
    return selected


def yolov8m_seg_choose_service_strategy_by_weight() -> Optional[Service]:
    """根据权重计算得分，选择合适的服务"""
    max_score = float('-inf')
    best_service = None

    services = get_services()
    service_queue_length_list = getServiceQueueLengthList()

    for service in services:
        if not service.healthy or service.device_type != "OrangePi 5B":
            continue

        # 获取设备类型和对应权重
        device_type = service.device_type
        weight_config = None

        nacos_config = get_nacos_config()

        for item in nacos_config.get("weight", []):
            if item["device"] == device_type:
                weight_config = item
                break

        if not weight_config:
            Logger.warning(f"未找到 {device_type} 的权重配置")
            continue

        yolov8n_pose_len = 10
        yolov8s_seg_len = 10
        yolov8m_seg_len = 10
        for service_queue_length in service_queue_length_list:
            if service_queue_length.healthy and service_queue_length.host == service.host and service_queue_length.port == service.port and service_queue_length.device_type == device_type:
                yolov8n_pose_len = service_queue_length.yolov8n_pose_len
                yolov8s_seg_len = service_queue_length.yolov8s_seg_len
                yolov8m_seg_len = service_queue_length.yolov8m_seg_len
                break

        # 权重系数
        cpu_weight = weight_config.get("cpu", 0)
        memory_weight = weight_config.get("memory", 0)
        npu_weight = weight_config.get("npu", 0)
        temp_weight = weight_config.get("temperature", 0)
        inference_time_weight = weight_config.get("yolov8m_seg_inference_time", 0)
        response_time_weight = weight_config.get("yolov8m_seg_response_time", 0)
        priority_base = weight_config.get("priority_base", 0)
        core_usage_weight = weight_config.get("core_usage", 0)
        aicpu_usage_weight = weight_config.get("aicpu_usage", 0)
        yolov8n_pose_len_weight = weight_config.get("yolov8n_pose_len", 0)
        yolov8s_seg_len_weight = weight_config.get("yolov8s_seg_len", 0)
        yolov8m_seg_len_weight = weight_config.get("yolov8m_seg_len", 0)
        task_cost_weight = weight_config.get("task_cost", 0)

        score = 0.0

        # cpu占用率
        if service.cpu_usage != -1:
            score += (100 - service.cpu_usage) * cpu_weight

        # 内存占用率
        if service.memory_usage != -1:
            score += (100 - service.memory_usage) * memory_weight

        # 温度
        if service.cpu_temp != -1:
            score += (100 - service.cpu_temp) * temp_weight

        # 优先级
        if service.priority != -1:
            score += service.priority * priority_base

        # 推理耗时
        score -= service.yolov8m_seg_inference_time * inference_time_weight

        # 平均响应耗时
        response_time_avg = (
            sum(service.yolov8m_seg_response_time_list) / len(service.yolov8m_seg_response_time_list)
            if service.yolov8m_seg_response_time_list else 0
        )

        score -= response_time_avg * response_time_weight

        # NPU 使用率取平均值
        npu_usage = (
            sum(service.npu_usages) / len(service.npu_usages)
            if service.npu_usages else -1
        )

        if npu_usage != -1:
            score += (100 - npu_usage) * npu_weight

        # AICore 使用率
        if service.aicore_usage != -1:
            score += (100 - service.aicore_usage) * core_usage_weight

        # AICPU 使用率
        if service.aicpu_usage != -1:
            score += (100 - service.aicpu_usage) * aicpu_usage_weight

        score -= yolov8n_pose_len * yolov8n_pose_len_weight
        score -= yolov8s_seg_len * yolov8s_seg_len_weight
        score -= yolov8m_seg_len * yolov8m_seg_len_weight

        score -= service.task_cost * task_cost_weight

        Logger.info(f"yolov8m_seg - 服务{service.host}:{service.port} 得分: {score:.2f}")

        if score > max_score:
            max_score = score
            best_service = service

    return best_service
