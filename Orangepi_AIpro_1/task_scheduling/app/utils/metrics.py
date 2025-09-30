import re
from .log import Logger
import subprocess

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
        # 执行命令并捕获输出
        result = subprocess.run(
            ["npu-smi", "info", "-t", "temp", "-i", "0", "-c", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # 解析输出以提取温度值
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            if "Temperature (C)" in line:
                # 提取冒号后的温度值并去除空格
                temp_str = line.split(":")[-1].strip()
                return float(temp_str)

        # 如果没有找到温度信息
        Logger.error("Failed to parse temperature from npu-smi output")
        return -1

    except subprocess.CalledProcessError as e:
        # 处理命令执行错误
        Logger.error(f"Failed to execute npu-smi command: {e.stderr}")
        return -1
    except Exception as e:
        # 处理其他异常
        Logger.error(f"An error occurred: {e}")
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


def get_aicore_aicpu_usage():
    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            ["npu-smi", "info", "-t", "usages", "-i", "0", "-c", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output_lines = result.stdout.splitlines()
        aicore_usage = -1
        aicpu_usage = -1

        for line in output_lines:
            if "Aicore Usage Rate(%)" in line:
                aicore_usage = int(line.split(":")[-1].strip())
            elif "Aicpu Usage Rate(%)" in line:
                aicpu_usage = int(line.split(":")[-1].strip())

        return {
            "aicore_usage": aicore_usage,
            "aicpu_usage": aicpu_usage
        }

    except subprocess.CalledProcessError as e:
        Logger.error(f"Failed to execute npu-smi command: {e.stderr}")
        return {"aicore_usage": -1, "aicpu_usage": -1}
    except Exception as e:
        Logger.error(f"An error occurred: {e}")
        return {"aicore_usage": -1, "aicpu_usage": -1}

