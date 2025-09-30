import re
from .log import Logger

class CPUUsage:
    def __init__(self, user=0, nice=0, system=0, idle=0, iowait=0, irq=0, softirq=0):
        self.user = user
        self.nice = nice
        self.system = system
        self.idle = idle
        self.iowait = iowait
        self.irq = irq
        self.softirq = softirq


def get_cpu_usage():
    usage = CPUUsage()
    try:
        with open("/proc/stat", "r") as fp:
            line = fp.readline()
            values = line.split()[1:8]
            usage.user, usage.nice, usage.system, usage.idle, usage.iowait, usage.irq, usage.softirq = map(int, values)
    except FileNotFoundError:
        Logger.error("Failed to open /proc/stat")
        exit(1)

    return usage


def calculate_cpu_usage(prev, curr):
    prev_total = prev.user + prev.nice + prev.system + prev.idle + prev.iowait + prev.irq + prev.softirq
    curr_total = curr.user + curr.nice + curr.system + curr.idle + curr.iowait + curr.irq + curr.softirq

    total_diff = curr_total - prev_total
    idle_diff = curr.idle - prev.idle

    if total_diff == 0:
        return 0.0

    return (total_diff - idle_diff) / total_diff * 100

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as fp:
            temp = int(fp.read().strip())
        # 将温度从毫摄氏度转换为摄氏度
        return temp / 1000
    except FileNotFoundError:
        Logger.error("Failed to read CPU temperature: File not found")
        return -1
    except ValueError:
        Logger.error("Failed to read CPU temperature: Invalid data")
        return -1
    except Exception as e:
        Logger.error(f"Failed to read CPU temperature: {e}")
        return -1


def get_memory_usage():
    try:
        with open("/proc/meminfo", "r") as fp:
            total_memory = 0
            available_memory = 0

            for line in fp:
                if line.startswith("MemTotal:"):
                    total_memory = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_memory = int(line.split()[1])

                # 如果已找到所需信息，则退出循环
                if total_memory > 0 and available_memory > 0:
                    break

        if total_memory == 0:
            return -1  # 无法读取到总内存

        # 计算使用率百分比
        usage = 100 - (available_memory * 100 / total_memory)
        return usage

    except FileNotFoundError:
        Logger.error("Failed to open /proc/meminfo: File not found")
        return -1
    except ValueError:
        Logger.error("Failed to read memory data: Invalid data format")
        return -1
    except Exception as e:
        Logger.error(f"Failed to read memory data: {e}")
        return -1


def get_npu_usage(core_count=3):
    # 初始化一个列表以保存每个核心的使用率
    npu_usage = [-1] * core_count  # 假设最多有 core_count 个核心

    try:
        # with open("/sys/kernel/debug/rknpu/load", "r") as fp:
        with open("/tmp/npu_load", "r") as fp:
            line = fp.readline().strip()
    except FileNotFoundError:
        Logger.error("Failed to read NPU load: File not found")
        return npu_usage
    except Exception as e:
        Logger.error(f"Failed to read NPU load: {e}")
        return npu_usage

    if not line:
        Logger.warning(f"[WARN] NPU data is empty!")
        return npu_usage

    pattern = r"^NPU load:\s*(Core\d+:\s*\d+%,\s*)+$"
    if not re.match(pattern, line):
        Logger.warning(f"[WARN] NPU data format abnormality: '{line}'")
        return npu_usage

    # 提取负载部分，格式类似于 "NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,"
    load_part = line.split(":", 1)[1].strip()  # 只取冒号后的部分
    tokens = load_part.split(",")[:-1]  # 去掉最后的空字符串部分

    try:
        for token in tokens:
            # 提取核心ID和使用率，假设格式为 "Core0:  0%" 等
            core_id, usage = token.strip().split(":")
            core_num = int(core_id.replace("Core", "").strip())
            usage_percent = int(usage.strip().replace("%", ""))
            if 0 <= core_num < core_count:
                npu_usage[core_num] = usage_percent
            else:
                Logger.error(f"Core number {core_num} is out of range.")
    except Exception as e:
        Logger.error(f"[ERROR] Failed to parse NPU load: {e}")

    return npu_usage  # 返回所有核心的使用率

