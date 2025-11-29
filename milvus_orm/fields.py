"""
Fields module for milvus_orm. Defines the Field classes used in Model definitions.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pymilvus import DataType


class Field(ABC):
    """Base class for all field types in milvus_orm."""

    MILVUS_TYPE: DataType

    def __init__(
        self,
        primary_key: bool = False,
        nullable: bool = False,
        default: Any = None,
        description: str = "",
        **kwargs,
    ):
        self.name = ""
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
        self.description = description
        self.kwargs = kwargs

        if self.primary_key and self.MILVUS_TYPE not in [
            DataType.INT64,
            DataType.VARCHAR,
        ]:
            raise ValueError(
                f"Primary key is not supported for field type {self.MILVUS_TYPE.name}"
            )

    @abstractmethod
    def to_milvus_type(self) -> dict:
        """Convert field definition to Milvus SDK format."""
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if the value is compatible with the field type."""
        pass


class IntegerField(Field):
    """Int64 field type."""

    MILVUS_TYPE = DataType.INT32

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "is_primary": self.primary_key,
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        return isinstance(value, int)


class BigIntField(IntegerField):
    """Int64 field type."""

    MILVUS_TYPE = DataType.INT64

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable or self.primary_key
        return isinstance(value, int)


class BooleanField(Field):
    """Boolean field type."""

    MILVUS_TYPE = DataType.BOOL

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        return isinstance(value, bool)


class CharField(Field):
    """Variable character string field type."""

    MILVUS_TYPE = DataType.VARCHAR

    def __init__(
        self,
        max_length: int = 65535,
        enable_analyzer: bool = False,
        analyzer_params: Optional[dict] = None,
        enable_match: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.enable_analyzer = enable_analyzer
        self.analyzer_params = analyzer_params
        self.enable_match = enable_match

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "max_length": self.max_length,
            "enable_analyzer": self.enable_analyzer,
            "analyzer_params": self.analyzer_params,
            "enable_match": self.enable_match,
            "is_primary": self.primary_key,
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        if not isinstance(value, str):
            return False
        return len(value) <= self.max_length


class UUIDField(CharField):
    """UUID field type."""

    def __init__(
        self,
        primary_key: bool = False,
        nullable: bool = False,
        default: Any = None,
        description: str = "",
        **kwargs,
    ):
        super().__init__(
            max_length=36,
            primary_key=primary_key,
            nullable=nullable,
            default=default,
            description=description,
            **kwargs,
        )

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable or self.primary_key
        if not isinstance(value, str):
            return False
        return len(value) <= self.max_length


class JsonField(Field):
    """JSON field type for storing structured data."""

    MILVUS_TYPE = DataType.JSON

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        # In practice, we'll rely on Milvus to validate JSON format
        return isinstance(value, (dict, list, str, int, float, bool))


class FloatField(Field):
    """Float field type."""

    MILVUS_TYPE = DataType.FLOAT

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        return isinstance(value, (int, float))


class VectorField(Field):
    """Base class for vector field types."""

    def __init__(self, index_type: Optional[str] = "AUTOINDEX", **kwargs):
        super().__init__(**kwargs)
        self.index_type = index_type


class DenseVectorField(VectorField):
    """Base class for dense vector field types."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "dim": self.dim,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        if not isinstance(value, list):
            return False
        if len(value) != self.dim:
            return False
        return all(isinstance(v, (int, float)) for v in value)


class FloatVectorField(DenseVectorField):
    """Dense float vector field type."""

    MILVUS_TYPE = DataType.FLOAT_VECTOR


class SparseFloatVectorField(Field):
    """Sparse float vector field type."""

    MILVUS_TYPE = DataType.SPARSE_FLOAT_VECTOR

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.MILVUS_TYPE,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        # Sparse vectors are typically represented as dictionaries
        # with indices as keys and values as floats
        if not isinstance(value, dict):
            return False
        return all(
            isinstance(k, int) and isinstance(v, (int, float)) for k, v in value.items()
        )
