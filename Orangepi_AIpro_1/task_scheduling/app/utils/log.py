from datetime import datetime

class Logger:
    """分级日志记录器"""
    @staticmethod
    def success(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ {msg}")

    @staticmethod
    def warning(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ {msg}")

    @staticmethod
    def error(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ❌ {msg}")

    @staticmethod
    def info(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ℹ️ {msg}")
