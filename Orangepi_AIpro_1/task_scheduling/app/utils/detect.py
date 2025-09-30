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

def detect_image_by_local(image: np.ndarray, timeout: float = 2.0, file_path: str = "./result/response_time/response_time.jsonl") -> Dict[str, Any]:
    """
    将 OpenCV 图像发送到本地 /infer 推理接口进行推理

    参数:
        image: 要推理的图像 (np.ndarray, BGR 格式)
        is_draw: 是否在返回结果中绘制关键点
        timeout: 请求超时时间（秒）
        file_path: 用于记录响应时间的文件路径（JSONL 格式）

    返回:
        推理结果（解析后的 JSON）
    """
    if image is None or not isinstance(image, np.ndarray):
        Logger.error("无效的图像输入")
        return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

    try:
        # 将图像编码为 JPEG 格式的字节流
        success, img_encoded = cv2.imencode(".jpg", image)
        if not success:
            Logger.error("图像编码失败")
            return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

        # 构建文件对象模拟 UploadFile
        files = {
            "image": ("image.jpg", BytesIO(img_encoded.tobytes()), "image/jpeg")
        }

        start_time = time.time()  # 新增：记录开始时间

        # 发送 POST 请求
        response = requests.post(
            url="http://localhost:8080/infer",
            files=files,
            timeout=timeout
        )

        latency = time.time() - start_time  # 新增：计算耗时

        # 检查响应状态码
        if response.status_code != 200:
            try:
                error_data = response.json()
                error_msg = error_data.get("msg", f"HTTP {response.status_code}")
            except Exception:
                error_msg = f"HTTP {response.status_code}"
            raise RuntimeError(f"推理服务异常: {error_msg}")

        # 解析 JSON 响应
        result = response.json()

        # 检查是否成功
        if result.get("code") != 200:
            raise RuntimeError(f"推理失败: {result.get('msg', '未知错误')}")

        # 确保文件路径存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 新增：将耗时信息写入文件（JSONL格式）
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "inference_latency": latency
        }

        # 追加写入 JSON 行到文件
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

        return result.get("data", {})

    except requests.exceptions.Timeout:
        Logger.error(f"本地推理请求超时（>{timeout}s）")
        return {"boxes": [], "classes": [], "scores": [], "keypoints": []}
    except requests.exceptions.RequestException as e:
        Logger.error(f"网络请求失败: {str(e)}")
        return {"boxes": [], "classes": [], "scores": [], "keypoints": []}
    except Exception as e:
        Logger.error(f"本地推理异常: {str(e)}")
        return {"boxes": [], "classes": [], "scores": [], "keypoints": []}


def choose_service_strategy_by_weight() -> Optional[Service]:
    """根据权重计算得分，选择合适的服务"""
    max_score = float('-inf')
    best_service = None

    services = get_services()
    service_queue_length_list = getServiceQueueLengthList()

    for service in services:
        if not service.healthy:
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
        inference_time_weight = weight_config.get("vehicle_inference_time", 0)
        response_time_weight = weight_config.get("vehicle_response_time", 0)
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
        score -= service.vehicle_inference_time * inference_time_weight

        # 平均响应耗时
        response_time_avg = (
            sum(service.vehicle_response_time_list) / len(service.vehicle_response_time_list)
            if service.vehicle_response_time_list else 0
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

        # 减去队列长度
        score -= yolov8n_pose_len * yolov8n_pose_len_weight
        score -= yolov8s_seg_len * yolov8s_seg_len_weight
        score -= yolov8m_seg_len * yolov8m_seg_len_weight

        score -= service.task_cost * task_cost_weight

        Logger.info(f"vehicle - 服务{service.host}:{service.port} 得分: {score:.2f}")

        if score > max_score:
            max_score = score
            best_service = service

    return best_service

def detect_image_by_weight(image: np.ndarray, timeout: float = 2.0, file_path: str = "./result/response_time/response_time.jsonl", max_retries: int = 1) -> Dict[str, Any]:
    """
    将 OpenCV 图像发送到得分最高的推理接口进行推理

    参数:
        image: 要推理的图像 (np.ndarray, BGR 格式)
        is_draw: 是否在返回结果中绘制关键点
        timeout: 请求超时时间（秒）
        file_path: 用于记录响应时间的文件路径（JSONL 格式）

    返回:
        推理结果（解析后的 JSON）
    """
    if image is None or not isinstance(image, np.ndarray):
        Logger.error("无效的图像输入")
        return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

    retries = 0

    while retries < max_retries:
        retries += 1
        try:
            service = choose_service_strategy_by_weight()

            if service is None:
                Logger.error("没有可用的推理服务")
                return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

            # 将图像编码为 JPEG 格式的字节流
            success, img_encoded = cv2.imencode(".jpg", image)
            if not success:
                Logger.error("图像编码失败")
                return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

            # 构建文件对象模拟 UploadFile
            files = {
                "image": ("image.jpg", BytesIO(img_encoded.tobytes()), "image/jpeg")
            }

            service.task_cost += 1.0

            start_time = time.time()  # 新增：记录开始时间

            # 发送 POST 请求
            response = requests.post(
                url=f"http://{service.host}:{service.port}/infer",
                files=files,
                timeout=timeout
            )

            latency = time.time() - start_time  # 新增：计算耗时

            service.task_cost -= 1.0

            # 检查响应状态码
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("msg", f"HTTP {response.status_code}")
                except Exception:
                    error_msg = f"HTTP {response.status_code}"
                raise RuntimeError(f"推理服务异常: {error_msg}")

            Logger.info(f"请求推理服务 {service.host}:{service.port} 成功")

            service.vehicle_response_time_list.append(latency)

            # 解析 JSON 响应
            result = response.json()

            # 检查是否成功
            if result.get("code") != 200:
                raise RuntimeError(f"推理失败: {result.get('msg', '未知错误')}")

            # 确保文件路径存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 新增：将耗时信息写入文件（JSONL格式）
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "inference_latency": latency
            }

            # 追加写入 JSON 行到文件
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

            return result.get("data", {})

        except requests.exceptions.Timeout:
            Logger.error(f"第 {retries} 次请求失败，推理请求超时（>{timeout}s）")
        except requests.exceptions.RequestException as e:
            Logger.error(f"第 {retries} 次请求失败，网络请求失败: {str(e)}")
        except Exception as e:
            Logger.error(f"第 {retries} 次请求失败，推理异常: {str(e)}")

    Logger.error(f"请求重试次数已达最大限制，执行本地调用")
    return detect_image_by_local(image, timeout, file_path)


_index_gen = count()
_choose_lock = threading.Lock()

def choose_service_strategy_by_circle() -> Optional[Service]:
    """
    使用线程安全的 itertools.count 实现轮询索引生成
    """
    services = get_services()
    if not services:
        Logger.error("没有可用的推理服务")
        return None

    healthy_services = [s for s in services if s.healthy and s.host and s.port]
    if not healthy_services:
        Logger.warning("没有健康的推理服务")
        return None

    with _choose_lock:
        index = next(_index_gen) % len(healthy_services)

    selected = healthy_services[index]
    Logger.info(f"使用轮询策略选择了服务: {selected.host}:{selected.port}")
    return selected


def detect_image_by_circle(image: np.ndarray, timeout: float = 2.0, file_path: str = "./result/response_time/response_time.jsonl", max_retries: int = 1) -> Dict[str, Any]:
    """
    轮询进行推理

    参数:
        image: 要推理的图像 (np.ndarray, BGR 格式)
        is_draw: 是否在返回结果中绘制关键点
        timeout: 请求超时时间（秒）
        file_path: 用于记录响应时间的文件路径（JSONL 格式）

    返回:
        推理结果（解析后的 JSON）
    """
    if image is None or not isinstance(image, np.ndarray):
        Logger.error("无效的图像输入")
        return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

    retries = 0

    while retries < max_retries:
        retries += 1
        try:
            service = choose_service_strategy_by_circle()

            if service is None:
                Logger.error("没有可用的推理服务")
                return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

            # 将图像编码为 JPEG 格式的字节流
            success, img_encoded = cv2.imencode(".jpg", image)
            if not success:
                Logger.error("图像编码失败")
                return {"boxes": [], "classes": [], "scores": [], "keypoints": []}

            # 构建文件对象模拟 UploadFile
            files = {
                "image": ("image.jpg", BytesIO(img_encoded.tobytes()), "image/jpeg")
            }

            start_time = time.time()  # 新增：记录开始时间

            # 发送 POST 请求
            response = requests.post(
                url=f"http://{service.host}:{service.port}/infer",
                files=files,
                timeout=timeout
            )

            latency = time.time() - start_time  # 新增：计算耗时

            # 检查响应状态码
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("msg", f"HTTP {response.status_code}")
                except Exception:
                    error_msg = f"HTTP {response.status_code}"
                raise RuntimeError(f"推理服务异常: {error_msg}")

            Logger.info(f"请求推理服务 {service.host}:{service.port} 成功")

            service.vehicle_response_time_list.append(latency)

            # 解析 JSON 响应
            result = response.json()

            # 检查是否成功
            if result.get("code") != 200:
                raise RuntimeError(f"推理失败: {result.get('msg', '未知错误')}")

            # 确保文件路径存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 新增：将耗时信息写入文件（JSONL格式）
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "inference_latency": latency
            }

            # 追加写入 JSON 行到文件
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')

            return result.get("data", {})

        except requests.exceptions.Timeout:
            Logger.error(f"第 {retries} 次请求失败，推理请求超时（>{timeout}s）")
        except requests.exceptions.RequestException as e:
            Logger.error(f"第 {retries} 次请求失败，网络请求失败: {str(e)}")
        except Exception as e:
            Logger.error(f"第 {retries} 次请求失败，推理异常: {str(e)}")

    Logger.error(f"请求重试次数已达最大限制，执行本地调用")
    return detect_image_by_local(image, timeout, file_path)



def detect_image(image: np.ndarray, timeout: float = 2.0, file_path: str = "./result/response_time/response_time.jsonl", max_retries: int = 1) -> Dict[str, Any]:
    """
    统一入口，方便切换策略
    :param image:
    :param timeout:
    :param file_path:
    :return:
    """
    # return detect_image_by_local(image=image, timeout=timeout, file_path=file_path)
    return detect_image_by_weight(image=image, timeout=timeout, file_path=file_path, max_retries=max_retries)
    # return detect_image_by_circle(image=image, timeout=timeout, file_path=file_path, max_retries=max_retries)



if __name__=="__main__":
    # 读取图像
    # image = cv2.imread("/home/orangepi/rockchip/rknn_toolkit_lite2/examples/task_scheduling/app/result/stitched_20250617_174445.jpg")
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    # 推理
    # result = detect_image(image)

    # draw(image, result["boxes"], result["scores"], result["classes"], result["keypoints"])

    # 保存结果图像到指定目录
    # result_dir = "/home/orangepi/rockchip/rknn_toolkit_lite2/examples/task_scheduling/app/result"
    # output_path = f"{result_dir}/detected_{cv2.getTickCount()}.jpg"
    # cv2.imwrite(output_path, image)
    # print(f"结果已保存至: {output_path}")

    # print(result)
