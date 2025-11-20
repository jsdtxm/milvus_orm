"""
milvus_orm - A Django-style ORM for Milvus vector database with async support.
"""

__version__ = "0.1.0"


# 导入客户端函数
from .client import connect, disconnect, get_client

# 导入字段类型
from .fields import (
    BigIntField,
    BooleanField,
    CharField,
    Field,
    FloatField,
    FloatVectorField,
    IntegerField,
    JsonField,
    SparseFloatVectorField,
)

# 导入模型类
from .models import Model

__all__ = [
    "Field",
    "BigIntField",
    "BooleanField",
    "CharField",
    "FloatField",
    "FloatVectorField",
    "IntegerField",
    "JsonField",
    "SparseFloatVectorField",
    "Model",
    "connect",
    "disconnect",
    "get_client",
]
