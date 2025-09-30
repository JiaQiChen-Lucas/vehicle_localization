import nacos
from flask import Flask, request, jsonify
import json
import time
import yaml
import threading
import numpy as np
import cv2
from ..utils import Logger, get_cpu_usage, calculate_cpu_usage, get_cpu_temp, get_memory_usage, get_npu_usage, RKNNPoolExecutor

app = Flask(__name__)

# nacos 配置
SERVER_ADDRESSES = "http://192.168.3.88:8848"
NAMESPACE = "8171f24f-7a40-414a-8144-6e0fd316f1e8"
USERNAME = "nacos"
PASSWORD = "nacos"

# 服务配置
SERVICE_NAME = "vehicleDetectService"
SERVICE_IP = "192.168.3.120"
SERVICE_PORT = 8080

# 配置项
CONFIG_DATA_ID = "vehicle_application"
CONFIG_GROUP = "DEFAULT_GROUP"

# 元数据
DEVICE_TYPE = "OrangePi 5B"
INFERENCE_TIME = 18   # 推理耗时
PRIORITY = 1        # 优先级


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

def update_metrics():
    """更新系统指标"""
    global metrics

    metrics["prev_usage"] = metrics["curr_usage"]
    metrics["curr_usage"] = get_cpu_usage()

    metrics["cpu_usage"] = calculate_cpu_usage(metrics["prev_usage"], metrics["curr_usage"])
    metrics["cpu_temp"] = get_cpu_temp()
    metrics["memory_usage"] = get_memory_usage()
    metrics["npu_usages"] = get_npu_usage()

    Logger.success(f"系统指标已更新: CPU={metrics['cpu_usage']:.2f}%, 温度={metrics['cpu_temp']:.2f}°C, 内存={metrics['memory_usage']:.2f}%, NPUs: {metrics['npu_usages']}")


def service_register():
    """注册服务到Nacos"""
    try:
        client.add_naming_instance(
            service_name=SERVICE_NAME,
            ip=SERVICE_IP,
            port=SERVICE_PORT,
            ephemeral=True,  # 临时实例
            healthy=True,
            heartbeat_interval=5,
            cluster_name='DEFAULT',
            group_name='DEFAULT_GROUP',
            metadata={
                "cpu_usage": "{0:.2f}".format(metrics["cpu_usage"]),
                "cpu_temp": "{0:.2f}".format(metrics["cpu_temp"]),
                "memory_usage": "{0:.2f}".format(metrics["memory_usage"]),
                "npu_usages": json.dumps(metrics["npu_usages"]),
                "device_type": DEVICE_TYPE,
                "inference_time": INFERENCE_TIME,
                "priority": PRIORITY
            }
        )
        Logger.success(f"服务注册成功: {SERVICE_NAME}@{SERVICE_IP}:{SERVICE_PORT}")
    except Exception as e:
        Logger.error(f"服务注册失败: {str(e)}")

def update_nacos_metadata():
    """更新Nacos元数据"""
    try:
        # 更新一次指标
        update_metrics()

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
                "inference_time": INFERENCE_TIME,
                "priority": PRIORITY
            }
        )
    except Exception as e:
        Logger.warning(f"元数据更新失败: {str(e)}")

def metrics_loop():
    # time.sleep(5)
    # 初始化CPU使用率
    metrics["curr_usage"] = get_cpu_usage()
    """指标更新循环"""
    while True:
        time.sleep(5)  # 每5秒更新一次
        try:
            update_nacos_metadata()
        except Exception as e:
            Logger.error(f"指标更新异常: {str(e)}")


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

# @app.route('/nacosConfig', methods=['GET'])
# def nacosConfig():
#     return json.dumps(nacos_config)

@app.post("/infer")
async def handle_request():
    try:
        # 检查是否有文件上传
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        # 检查文件类型
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400

        # 验证文件是否为图片
        try:
            # image = Image.open(io.BytesIO(file.read())).convert('RGB')
            file_bytes = np.frombuffer(file.read(), np.uint8)  # 将文件内容转为字节数组
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # 使用 OpenCV 解码为图像
        except Exception as e:
            return jsonify({"error": "Uploaded file is not a valid image!"}), 400

        # 提交推理任务
        future = executor.submit(image, is_request=True)

        # 等待结果
        try:
            result = future.result(timeout=0.5)
            return jsonify({"status": "success", "result": result.tolist()}), 200
        except TimeoutError:
            return {"status": "timeout"}

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # 创建执行器
    executor = RKNNPoolExecutor(rknn_model=RKNN_MODEL, max_workers=6, cores=3)

    # no auth mode
    # client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)
    # auth mode
    client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, username=USERNAME, password=PASSWORD)

    # threading.Timer(5, service_register).start()
    threading.Thread(target=service_register, daemon=True).start()
    # 启动指标更新线程
    # 设置该线程为守护线程（后台线程），主程序退出时它会自动结束
    threading.Thread(target=metrics_loop, daemon=True).start()

    # 启动配置监听线程
    threading.Thread(target=start_config_watch, daemon=True).start()

    app.run(host='0.0.0.0', port=8080, debug=False)
