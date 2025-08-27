# constants.py
from enum import Enum, unique


@unique
class HttpStatus(Enum):
    """
    定义一组HTTP状态码常量。
    @unique 装饰器确保所有成员的值都是唯一的。
    """
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500
@unique
class MCPConstant(Enum):
    """
    静态枚举常量。
    @unique 装饰器确保所有成员的值都是唯一的。
    """
    MCP_SERVER_URL = "http://127.0.0.1:8000"  # 您的MCP服务地址
