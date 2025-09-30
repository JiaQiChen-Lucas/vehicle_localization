from pathlib import Path


class Settings:
    # 项目基础配置
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DEBUG = True

    # 日志配置
    LOG_ROTATION = "00:00"  # 每天午夜轮转
    LOG_RETENTION = "30 days"  # 保留30天
    LOG_COMPRESSION = "zip"  # 压缩格式


settings = Settings()
