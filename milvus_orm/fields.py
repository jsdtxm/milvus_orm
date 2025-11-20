"""
Fields module for milvus_orm. Defines the Field classes used in Model definitions.
"""

from abc import ABC, abstractmethod
from typing import Any


class Field(ABC):
    """Base class for all field types in milvus_orm."""

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

    @abstractmethod
    def to_milvus_type(self) -> dict:
        """Convert field definition to Milvus SDK format."""
        pass

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if the value is compatible with the field type."""
        pass


class INT64(Field):
    """Int64 field type."""

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "data_type": "Int64",
            "is_primary_key": self.primary_key,
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        return isinstance(value, int)


class VARCHAR(Field):
    """Variable character string field type."""

    def __init__(
        self,
        max_length: int = 65535,
        enable_analyzer: bool = False,
        enable_match: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.enable_analyzer = enable_analyzer
        self.enable_match = enable_match

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "data_type": "VarChar",
            "max_length": self.max_length,
            "enable_analyzer": self.enable_analyzer,
            "enable_match": self.enable_match,
            "is_primary_key": self.primary_key,
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


class JSON(Field):
    """JSON field type for storing structured data."""

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "data_type": "JSON",
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        # In practice, we'll rely on Milvus to validate JSON format
        return isinstance(value, (dict, list, str, int, float, bool))


class FLOAT(Field):
    """Float field type."""

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "data_type": "Float",
            "nullable": self.nullable,
            "description": self.description,
            **self.kwargs,
        }

    def validate(self, value: Any) -> bool:
        if value is None:
            return self.nullable
        return isinstance(value, (int, float))


class FLOAT_VECTOR(Field):
    """Dense float vector field type."""

    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "data_type": "FloatVector",
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


class SPARSE_FLOAT_VECTOR(Field):
    """Sparse float vector field type."""

    def to_milvus_type(self) -> dict:
        return {
            "name": self.name,
            "data_type": "SparseFloatVector",
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
