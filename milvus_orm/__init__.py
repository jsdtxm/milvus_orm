"""
milvus_orm - A Django-style ORM for Milvus vector database with async support.
"""

__version__ = "0.1.0"


# 导入客户端函数
from .client import connect, disconnect, get_client

# 导入字段类型
from .fields import (
    FLOAT,
    FLOAT_VECTOR,
    INT64,
    JSON,
    SPARSE_FLOAT_VECTOR,
    VARCHAR,
    Field,
)

# 导入模型类
from .models import Model

__all__ = [
    "Field",
    "INT64",
    "VARCHAR",
    "JSON",
    "FLOAT",
    "FLOAT_VECTOR",
    "SPARSE_FLOAT_VECTOR",
    "Model",
    "connect",
    "disconnect",
    "get_client",
]
