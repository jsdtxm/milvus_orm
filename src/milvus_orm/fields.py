"""
字段类型定义，支持各种Milvus数据类型
"""

from typing import Any, List, Optional, Dict, Union
from enum import Enum


class FieldType(Enum):
    """Milvus字段类型枚举"""
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    BOOL = "BOOL"
    VARCHAR = "VARCHAR"
    BINARY_VECTOR = "BINARY_VECTOR"
    FLOAT_VECTOR = "FLOAT_VECTOR"


class Field:
    """字段基类"""
    
    def __init__(
        self,
        primary_key: bool = False,
        auto_id: bool = False,
        default: Any = None,
        description: str = "",
        **kwargs
    ):
        self.primary_key = primary_key
        self.auto_id = auto_id
        self.default = default
        self.description = description
        self.name = None  # 将在模型初始化时设置
        
    def to_milvus_schema(self) -> Dict[str, Any]:
        """转换为Milvus字段定义"""
        raise NotImplementedError("子类必须实现此方法")
    
    def validate(self, value: Any) -> Any:
        """验证字段值"""
        return value


class IntField(Field):
    """整数字段"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field_type = FieldType.INT64
    
    def to_milvus_schema(self) -> Dict[str, Any]:
        schema = {
            "name": self.name,
            "dtype": self.field_type.value,
            "is_primary": self.primary_key,
            "auto_id": self.auto_id,
            "description": self.description or ""
        }
        return schema
    
    def validate(self, value: Any) -> int:
        if value is None:
            return self.default
        return int(value)


class FloatField(Field):
    """浮点数字段"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field_type = FieldType.FLOAT
    
    def to_milvus_schema(self) -> Dict[str, Any]:
        schema = {
            "name": self.name,
            "dtype": self.field_type.value,
            "is_primary": self.primary_key,
            "auto_id": self.auto_id,
            "description": self.description or ""
        }
        return schema
    
    def validate(self, value: Any) -> float:
        if value is None:
            return self.default
        return float(value)


class CharField(Field):
    """字符串字段"""
    
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__(**kwargs)
        self.field_type = FieldType.VARCHAR
        self.max_length = max_length
    
    def to_milvus_schema(self) -> Dict[str, Any]:
        schema = {
            "name": self.name,
            "dtype": self.field_type.value,
            "is_primary": self.primary_key,
            "auto_id": self.auto_id,
            "max_length": self.max_length,
            "description": self.description or ""
        }
        return schema
    
    def validate(self, value: Any) -> str:
        if value is None:
            return self.default or ""
        value = str(value)
        if len(value) > self.max_length:
            raise ValueError(f"String length exceeds max_length of {self.max_length}")
        return value


class VectorField(Field):
    """向量字段"""
    
    def __init__(self, dim: int, metric_type: str = "L2", **kwargs):
        super().__init__(**kwargs)
        self.field_type = FieldType.FLOAT_VECTOR
        self.dim = dim
        self.metric_type = metric_type
        self.index_type = "IVF_FLAT"
        self.index_params = {"nlist": 128}
        
    def to_milvus_schema(self) -> Dict[str, Any]:
        schema = {
            "name": self.name,
            "dtype": self.field_type.value,
            "dim": self.dim,
            "description": self.description or ""
        }
        return schema
    
    def validate(self, value: Any) -> List[float]:
        if value is None:
            return self.default
        
        if not isinstance(value, (list, tuple)):
            raise ValueError("Vector field requires list or tuple")
        
        if len(value) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(value)}")
        
        return [float(x) for x in value]
    
    def distance(self, vector: List[float]) -> "VectorDistance":
        """创建向量距离表达式"""
        return VectorDistance(self, vector)


class VectorDistance:
    """向量距离表达式"""
    
    def __init__(self, field: VectorField, vector: List[float]):
        self.field = field
        self.vector = vector
    
    def __lt__(self, other: float) -> str:
        """小于比较"""
        return f"{self.field.name} < {other}"
    
    def __le__(self, other: float) -> str:
        """小于等于比较"""
        return f"{self.field.name} <= {other}"
    
    def __gt__(self, other: float) -> str:
        """大于比较"""
        return f"{self.field.name} > {other}"
    
    def __ge__(self, other: float) -> str:
        """大于等于比较"""
        return f"{self.field.name} >= {other}"


class BooleanField(Field):
    """布尔字段"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field_type = FieldType.BOOL
    
    def to_milvus_schema(self) -> Dict[str, Any]:
        schema = {
            "name": self.name,
            "dtype": self.field_type.value,
            "description": self.description or ""
        }
        return schema
    
    def validate(self, value: Any) -> bool:
        if value is None:
            return self.default or False
        return bool(value)