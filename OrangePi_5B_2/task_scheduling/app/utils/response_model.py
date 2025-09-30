from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, conint

# 定义泛型类型变量
T = TypeVar('T')

class ResponseModel(BaseModel, Generic[T]):
    """统一API响应模型"""
    code: int = 200        # 业务状态码，默认200
    success: bool = True              # 操作是否成功
    data: Optional[T] = None          # 响应数据
    msg: str = "操作成功"           # 响应消息

    @classmethod
    def success_response(cls, data: T = None, msg: str = "操作成功"):
        """创建成功响应"""
        return cls(code=200, success=True, data=data, msg=msg)

    @classmethod
    def error_response(cls, code: int, data: T = None, msg: str = "操作失败"):
        """创建错误响应"""
        return cls(code=code, success=False, data=data, msg=msg)
