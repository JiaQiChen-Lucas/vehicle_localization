from collections import deque
from datetime import datetime
from typing import List
import nacos
import json
import time
import yaml
import os
from .log import Logger
from .metrics import get_cpu_usage, calculate_cpu_usage, get_cpu_temp, get_memory_usage, get_npu_usage

# nacos 配置
SERVER_ADDRESSES = "http://192.168.3.101:8848"
NAMESPACE = "8171f24f-7a40-414a-8144-6e0fd316f1e8"
USERNAME = "nacos"
PASSWORD = "nacos"

# 服务配置
SERVICE_NAME = "vehicleDetectService"
SERVICE_IP = "192.168.3.124"
SERVICE_PORT = 8080

# 配置项
CONFIG_DATA_ID = "vehicle_application"
CONFIG_GROUP = "DEFAULT_GROUP"

# 元数据
DEVICE_TYPE = "OrangePi 5B"
VEHICLE_INFERENCE_TIME = 18   # 推理耗时
YOLOV8S_SEG_INFERENCE_TIME = 100
YOLOV8M_SEG_INFERENCE_TIME = 260
PRIORITY = 1        # 优先级

# 间隔
SAMPLE_INTERVAL: float = 1.0


# 全局变量（使用字典更安全）
metrics = {
    "prev_usage": 0,    # 前一次CPU指标
    "curr_usage": 0,    # 当前CPU指标
    "cpu_usage": 0,     # CPU利用率
    "cpu_temp": 0,      # CPU温度
    "memory_usage": 0,  # 内存利用率
    "npu_usages": []     # NPU利用率
}

RKNN_MODEL = "./model/best_Pose_RepVGG_ReLU_train_opset19_deploy.rknn"

# 全局配置存储
nacos_config = {}

# no auth mode
# client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)
# auth mode
client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, username=USERNAME, password=PASSWORD)

services = []

def get_available_inference_instances() -> list:
    """
    从 Nacos 获取指定服务的所有可用实例信息

    参数:
        service_name: 服务名称
        group_name: 分组名称
        cluster_name: 集群名称

    返回:
        list: 可用实例列表，每个元素为实例信息字典
    """
    try:
        # 调用 Nacos SDK 获取服务实例列表
        instances = client.list_naming_instance(
            service_name=SERVICE_NAME,
            clusters="DEFAULT",  # 使用 clusters 参数
            namespace_id=NAMESPACE,  # 使用全局 NAMESPACE
            group_name="DEFAULT_GROUP",  # 使用传入的 group_name
            healthy_only=True  # 仅获取健康实例
        )

        if not instances:
            Logger.warning(f"未找到服务 {SERVICE_NAME} 的可用实例")
            return []

        # print(instances)

        hosts = instances.get("hosts", [])
        if not hosts:
            Logger.warning(f"服务 {SERVICE_NAME} 没有可用实例或返回为空")
            return []

        instance_list = []
        for host in hosts:
            if not isinstance(host, dict):
                Logger.warning(f"跳过非字典格式的实例: {host}")
                continue

            instance_info = {
                "ip": host.get("ip"),
                "port": host.get("port"),
                "metadata": host.get("metadata", {}),
                "weight": host.get("weight"),
                "healthy": host.get("healthy"),
                "instance_id": host.get("instanceId")
            }
            instance_list.append(instance_info)

        Logger.info(f"成功获取 {len(instance_list)} 个 {SERVICE_NAME} 实例")
        return instance_list

    except Exception as e:
        Logger.error(f"获取 Nacos 实例信息失败: {str(e)}")
        return []


def update_metrics(file_path: str):
    """更新系统指标"""
    global metrics

    try:
        metrics["prev_usage"] = metrics["curr_usage"]
        metrics["curr_usage"] = get_cpu_usage()

        metrics["cpu_usage"] = calculate_cpu_usage(metrics["prev_usage"], metrics["curr_usage"])
        metrics["cpu_temp"] = get_cpu_temp()
        metrics["memory_usage"] = get_memory_usage()
        metrics["npu_usages"] = get_npu_usage()

        Logger.success(f"系统指标已更新: CPU={metrics['cpu_usage']:.2f}%, 温度={metrics['cpu_temp']:.2f}°C, 内存={metrics['memory_usage']:.2f}%, NPUs: {metrics['npu_usages']}")

        # 将metrics追加到file_path中
        # 添加时间戳
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "cpu_usage": metrics["cpu_usage"],
            "cpu_temp": metrics["cpu_temp"],
            "memory_usage": metrics["memory_usage"],
            "npu_usages": metrics["npu_usages"]
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 追加写入 JSON 行到文件
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

    except Exception as e:
        Logger.error(f"更新或保存指标失败: {str(e)}")

def service_register():
    """注册服务到Nacos"""
    try:
        client.add_naming_instance(
            service_name=SERVICE_NAME,
            ip=SERVICE_IP,
            port=SERVICE_PORT,
            ephemeral=True,  # 临时实例
            healthy=True,
            heartbeat_interval=SAMPLE_INTERVAL,
            cluster_name='DEFAULT',
            group_name='DEFAULT_GROUP',
            metadata={
                "cpu_usage": "{0:.2f}".format(metrics["cpu_usage"]),
                "cpu_temp": "{0:.2f}".format(metrics["cpu_temp"]),
                "memory_usage": "{0:.2f}".format(metrics["memory_usage"]),
                "npu_usages": json.dumps(metrics["npu_usages"]),
                "device_type": DEVICE_TYPE,
                "vehicle_inference_time": VEHICLE_INFERENCE_TIME,
                "yolov8s_seg_inference_time": YOLOV8S_SEG_INFERENCE_TIME,
                "yolov8m_seg_inference_time": YOLOV8M_SEG_INFERENCE_TIME,
                "priority": PRIORITY
            }
        )
        Logger.success(f"服务注册成功: {SERVICE_NAME}@{SERVICE_IP}:{SERVICE_PORT}")
    except Exception as e:
        Logger.error(f"服务注册失败: {str(e)}")

def update_nacos_metadata(file_path: str):
    """更新Nacos元数据"""
    try:
        # 更新一次指标
        update_metrics(file_path)

        client.modify_naming_instance(
            service_name=SERVICE_NAME,
            ip=SERVICE_IP,
            port=SERVICE_PORT,
            ephemeral=True,  # 临时实例
            cluster_name='DEFAULT',
            group_name='DEFAULT_GROUP',
            metadata={
                "cpu_usage": "{0:.2f}".format(metrics["cpu_usage"]),
                "cpu_temp": "{0:.2f}".format(metrics["cpu_temp"]),
                "memory_usage": "{0:.2f}".format(metrics["memory_usage"]),
                "npu_usages": json.dumps(metrics["npu_usages"]),
                "device_type": DEVICE_TYPE,
                "vehicle_inference_time": VEHICLE_INFERENCE_TIME,
                "yolov8s_seg_inference_time": YOLOV8S_SEG_INFERENCE_TIME,
                "yolov8m_seg_inference_time": YOLOV8M_SEG_INFERENCE_TIME,
                "priority": PRIORITY
            }
        )
    except Exception as e:
        Logger.warning(f"元数据更新失败: {str(e)}")

def get_config_from_nacos(content=None):
    """从Nacos获取配置"""
    try:
        config = content
        if (content is None):
            config = client.get_config(
                data_id=CONFIG_DATA_ID,
                group=CONFIG_GROUP
            )

        if not config:
            Logger.warning("配置为空，请检查Nacos是否存在该配置")
            return {}

        # 尝试YAML解析
        try:
            return yaml.safe_load(config)
        except yaml.YAMLError:
            # 如果YAML解析失败，尝试JSON解析
            try:
                return json.loads(config)
            except json.JSONDecodeError:
                Logger.warning(f"配置不是有效的YAML或JSON，原始内容:\n{config[:200]}...")
                return {}
    except Exception as e:
        Logger.error(f"获取配置失败: {str(e)}")
        return {}

def update_nacos_config(content=None):
    """更新全局配置"""
    global nacos_config
    new_config = get_config_from_nacos(content)
    if new_config:
        nacos_config = new_config
        Logger.success(f"配置更新成功: {nacos_config}")

def config_listener(event):
    """配置变更监听器"""
    Logger.success("检测到配置变更，正在更新...")
    update_nacos_config(content=event["raw_content"])

def start_config_watch():
    # time.sleep(10)

    # 初始加载配置
    update_nacos_config()

    """启动配置监听"""
    client.add_config_watcher(
        data_id=CONFIG_DATA_ID,
        group=CONFIG_GROUP,
        cb=config_listener
    )
    Logger.success("配置监听已启动")


class Service:
    def __init__(self, host: str, port: str, healthy: bool, meta: dict):
        self.host = host
        self.port = port
        self.healthy = healthy

        # 使用长度为 3 的队列保存最近三次推理接口响应时间（单位：秒）
        self.vehicle_response_time_list = deque(maxlen=3)
        self.yolov8s_seg_response_time_list = deque(maxlen=3)
        self.yolov8m_seg_response_time_list = deque(maxlen=3)

        self.task_cost = 0.0   # 当前任务数造成的损失（每安排一个任务，就增加损失，结束一个任务，减少损失）

        # 字符串转 float
        self.cpu_usage = float(meta.get("cpu_usage", "-1"))
        self.memory_usage = float(meta.get("memory_usage", "-1"))
        self.cpu_temp = float(meta.get("cpu_temp", "-1"))
        self.vehicle_inference_time = float(meta.get("vehicle_inference_time", "-1"))
        self.yolov8s_seg_inference_time = float(meta.get("yolov8s_seg_inference_time", "-1"))
        self.yolov8m_seg_inference_time = float(meta.get("yolov8m_seg_inference_time", "-1"))

        self.device_type = meta.get("device_type", "other")
        self.priority = float(meta.get("priority", "-1"))

        # 特殊字段处理
        npu_usages_str = meta.get("npu_usages")
        try:
            self.npu_usages = json.loads(npu_usages_str) if npu_usages_str else []
        except json.JSONDecodeError:
            self.npu_usages = []

        self.aicore_usage = float(meta.get("aicore_usage", "-1"))
        self.aicpu_usage = float(meta.get("aicpu_usage", "-1"))

    def update_metrics(self, meta: dict):
        # 字符串转 float
        self.cpu_usage = float(meta.get("cpu_usage", "-1"))
        self.memory_usage = float(meta.get("memory_usage", "-1"))
        self.cpu_temp = float(meta.get("cpu_temp", "-1"))
        self.vehicle_inference_time = float(meta.get("vehicle_inference_time", "-1"))
        self.yolov8s_seg_inference_time = float(meta.get("yolov8s_seg_inference_time", "-1"))
        self.yolov8m_seg_inference_time = float(meta.get("yolov8m_seg_inference_time", "-1"))

        self.device_type = meta.get("device_type", "other")
        self.priority = float(meta.get("priority", "-1"))

        # 特殊字段处理
        npu_usages_str = meta.get("npu_usages")
        try:
            self.npu_usages = json.loads(npu_usages_str) if npu_usages_str else []
        except json.JSONDecodeError:
            self.npu_usages = []

        self.aicore_usage = float(meta.get("aicore_usage", "-1"))
        self.aicpu_usage = float(meta.get("aicpu_usage", "-1"))

    def print_info(self):
        Logger.info(f"服务实例: {self.host}:{self.port}, "
                    f"Healthy: {self.healthy}, "
                    f"vehicle_response_time_list: {list(self.vehicle_response_time_list)}, "
                    f"yolov8s_seg_response_time_list: {list(self.yolov8s_seg_response_time_list)}, "
                    f"yolov8m_seg_response_time_list: {list(self.yolov8m_seg_response_time_list)}, "
                    f"cpu_usage: {self.cpu_usage}, "
                    f"memory_usage: {self.memory_usage}, "
                    f"cpu_temp: {self.cpu_temp}, "
                    f"vehicle_inference_time: {self.vehicle_inference_time}, "
                    f"yolov8s_seg_inference_time: {self.yolov8s_seg_inference_time}, "
                    f"yolov8m_seg_inference_time: {self.yolov8m_seg_inference_time}, "
                    f"device_type: {self.device_type}, "
                    f"priority: {self.priority}, "
                    f"npu_usages: {self.npu_usages}, "
                    f"aicore_usage: {self.aicore_usage}, "
                    f"aicpu_usage: {self.aicpu_usage}, ")

def update_service_metrics():
    """
    更新服务实例指标
    :return:
    """
    global services

    instances = get_available_inference_instances()

    for service in services:
        service.healthy = False

    for instance in instances:
        host = instance['ip']
        port = str(instance['port'])
        meta = instance['metadata']
        healthy = instance['healthy']

        found = False
        for service in services:
            if service.host == host and service.port == port:
                service.healthy = healthy
                service.update_metrics(meta)
                found = True
                break

        if not found:
            new_service = Service(host=host, port=port, healthy=healthy, meta=meta)
            services.append(new_service)

    for service in services:
        service.print_info()


def metrics_loop(dir_path="./result/metrics"):
    # time.sleep(5)

    # 创建目录（如果不存在）
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 构造文件名（以当前时间命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{timestamp}.jsonl"
    file_path = os.path.join(dir_path, filename)

    global metrics

    # 初始化CPU使用率
    metrics["curr_usage"] = get_cpu_usage()
    """指标更新循环"""
    while True:
        time.sleep(SAMPLE_INTERVAL)  # 每5秒更新一次
        # 更新nacos中的元数据
        try:
            update_nacos_metadata(file_path)
        except Exception as e:
            Logger.error(f"指标更新异常: {str(e)}")

        # 更新服务指标
        try:
            update_service_metrics()
        except Exception as e:
            Logger.error(f"指标更新异常: {str(e)}")

def get_nacos_config():
    """
    获取全局配置
    :return:
    """
    return nacos_config

def get_services() -> List[Service]:
    """
    获取服务实例列表
    :return:
    """
    return services

def start_nacos(dir_path: str):
    # 注册服务
    service_register()

    # 加载并监听配置
    start_config_watch()

    # 循环更新指标
    metrics_loop(dir_path)
