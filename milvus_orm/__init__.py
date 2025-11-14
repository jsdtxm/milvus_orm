"""
milvus_orm - A Django-style ORM for Milvus vector database with async support.
"""

__version__ = "0.1.0"

# 导入字段类型
try:
    from milvus_orm.fields import (
        Field,
        INT64,
        VARCHAR,
        JSON,
        FLOAT,
        FLOAT_VECTOR,
        SPARSE_FLOAT_VECTOR,
    )
except ImportError:
    # 尝试相对导入
    from .fields import (
        Field,
        INT64,
        VARCHAR,
        JSON,
        FLOAT,
        FLOAT_VECTOR,
        SPARSE_FLOAT_VECTOR,
    )

# 导入模型类
try:
    from milvus_orm.models import Model
except ImportError:
    from .models import Model

# 导入客户端函数
try:
    from milvus_orm.client import connect, disconnect, get_client
except ImportError:
    from .client import connect, disconnect, get_client

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