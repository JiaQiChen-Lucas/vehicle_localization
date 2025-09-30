import logging
import os
import sys
from pathlib import Path

from loguru import logger

from .config import settings


class InterceptHandler(logging.Handler):
    """
    日志拦截处理器：将所有 Python 标准日志重定向到 Loguru （用于处理uvicorn / fastapi 等自带的日志）

    工作原理：
    1. 继承自 logging.Handler
    2. 重写 emit 方法处理日志记录
    3. 将标准库日志转换为 Loguru 格式
    """

    def emit(self, record: logging.LogRecord) -> None:
        # 尝试获取日志级别名称
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # 获取调用帧信息
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # 使用 Loguru 记录日志
        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage()
        )


def setup_logging():
    """
    配置日志系统

    功能：
    1. 控制台彩色输出
    2. 文件日志轮转
    3. 错误日志单独存储
    4. 异步日志记录
    """
    # 步骤1：移除默认处理器
    logger.configure(extra={"request_id": ''})  # Default values 否则会报错
    logger.remove()

    # 步骤2：定义日志格式
    log_format = (
        # 时间信息 + UUID
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | {extra[request_id]} | "
        # "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        # "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> {correlation_id} | "
        # 日志级别，居中对齐
        "<level>{level: ^8}</level> | "
        # 进程和线程信息
        "process [<cyan>{process}</cyan>]:<cyan>{thread}</cyan> | "
        # 文件、函数和行号
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        # 日志消息
        "<level>{message}</level>"
    )

    # 步骤3：配置控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level="DEBUG" if settings.DEBUG else "INFO",
        # enqueue=True,  # 启用异步写入
        backtrace=False,  # 显示完整的异常回溯
        # diagnose=True,  # 显示变量值等诊断信息
        # colorize=True,  # 启用彩色输出
        # filter=correlation_id_filter

    )

    # 步骤4：创建日志目录
    log_dir = Path(settings.BASE_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 步骤5：配置常规日志文件
    logger.add(
        str(Path(log_dir, "log.log")),
        format=log_format,
        level="INFO",
        rotation="10 MB",
        retention="1000 days",
        encoding="utf-8",
        enqueue=True,  # 异步写入
        # filter=correlation_id_filter

    )

    # 步骤6：配置错误日志文件
    logger.add(
        str(Path(log_dir, "error.log")),
        format=log_format,
        level="ERROR",
        rotation="10 MB",
        retention="100 week",
        encoding="utf-8",
        enqueue=True,  # 异步写入
        # filter=correlation_id_filter
    )

    # 步骤7：配置 标准库日志 / 第三方库日志
    logger_name_list = [name for name in logging.root.manager.loggerDict]
    for logger_name in logger_name_list:
        _logger = logging.getLogger(logger_name)
        _logger.setLevel(logging.INFO)
        _logger.handlers = []
        if '.' not in logger_name:
            _logger.addHandler(InterceptHandler())

    # 禁用 Uvicorn 的访问日志
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)