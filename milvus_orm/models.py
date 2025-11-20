"""
Models module for milvus_orm. Defines the Model base class and related functionality.
"""

import inspect
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar

from .client import ensure_connection
from .fields import Field

if TYPE_CHECKING:
    from .query import QuerySet

M = TypeVar("M", bound="Model")


class ModelMeta(type):
    """Metaclass for Model that processes fields and creates schema."""

    def __new__(mcs, name: str, bases: tuple, attrs: dict):
        # Skip processing for Model base class
        if name == "Model" and bases == (object,):
            return super().__new__(mcs, name, bases, attrs)

        # Collect fields from class attributes
        fields = {}
        primary_key_field = None

        for key, value in attrs.items():
            if isinstance(value, Field):
                value.name = key
                fields[key] = value
                if value.primary_key:
                    primary_key_field = key

        # If no primary key is defined, add a default one
        if not primary_key_field:
            from .fields import INT64

            id_field = INT64(primary_key=True, auto_id=True)
            id_field.name = "id"
            fields["id"] = id_field
            attrs["id"] = id_field
            primary_key_field = "id"

        # Add schema metadata to the class
        attrs["_fields"] = fields
        attrs["_primary_key_field"] = primary_key_field

        # Determine collection name (use class name if not specified)
        collection_name = attrs.get("collection_name", name.lower())
        attrs["collection_name"] = collection_name

        # Create the class
        return super().__new__(mcs, name, bases, attrs)


class Model(object, metaclass=ModelMeta):
    """Base class for all milvus_orm models."""

    # Will be set by metaclass
    _fields: Dict[str, Field] = {}
    _primary_key_field: str = "id"
    collection_name: str = ""

    def __init__(self, **kwargs):
        """Initialize a model instance with field values."""
        # Validate and set field values
        for field_name, field in self._fields.items():
            value = kwargs.get(field_name, field.default)
            if not field.validate(value):
                raise ValueError(f"Invalid value for field '{field_name}': {value}")
            setattr(self, field_name, value)

        # Store any extra fields in dynamic field if enabled
        self._extra_fields = {k: v for k, v in kwargs.items() if k not in self._fields}

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        data = {}
        for field_name in self._fields:
            value = getattr(self, field_name, None)
            if value is not None:
                data[field_name] = value

        # Add extra fields for dynamic schema
        if hasattr(self, "_extra_fields") and self._extra_fields:
            data.update(self._extra_fields)

        return data

    @classmethod
    def _get_schema(cls) -> Dict[str, Any]:
        """Generate Milvus schema from model fields."""
        fields = []
        for field_name, field in cls._fields.items():
            field_dict = field.to_milvus_type()
            fields.append(field_dict)

        schema = {
            "collection_name": cls.collection_name,
            "fields": fields,
            "enable_dynamic_field": True,
        }
        return schema

    @classmethod
    async def create_collection(cls) -> bool:
        """Create collection in Milvus based on model schema."""
        client = await ensure_connection()
        schema = cls._get_schema()

        # Check if collection already exists
        if await client.has_collection(collection_name=cls.collection_name):
            return False

        # Create the collection
        await client.create_collection(
            collection_name=cls.collection_name,
            fields=schema["fields"],
            enable_dynamic_field=schema["enable_dynamic_field"],
        )
        return True

    @classmethod
    async def drop_collection(cls) -> bool:
        """Drop collection from Milvus."""
        client = await ensure_connection()

        # Check if collection exists
        if not await client.has_collection(collection_name=cls.collection_name):
            return False

        # Drop the collection
        await client.drop_collection(collection_name=cls.collection_name)
        return True

    @classmethod
    async def bulk_create(cls, instances: List["Model"]) -> int:
        """Bulk create multiple model instances.

        Args:
            instances: List of model instances to create

        Returns:
            Number of instances created
        """
        if not instances:
            return 0

        client = await ensure_connection()

        # Check if collection exists, create if not
        if not await client.has_collection(collection_name=cls.collection_name):
            await cls.create_collection()

        # Convert instances to dictionaries
        data = [instance.to_dict() for instance in instances]

        # Insert data in bulk
        result = await client.insert(collection_name=cls.collection_name, data=data)

        # Update primary keys if auto_id is enabled
        primary_key = cls._primary_key_field
        primary_key_values = result.get("primary_keys", [])

        if primary_key_values:
            for i, instance in enumerate(instances):
                if i < len(primary_key_values) and (
                    not hasattr(instance, primary_key)
                    or getattr(instance, primary_key) is None
                ):
                    setattr(instance, primary_key, primary_key_values[i])

        return result.get("insert_count", 0)

    @classmethod
    def objects(cls: Type[M]) -> "QuerySet[M]":
        """Return a QuerySet for the model."""
        from .query import QuerySet

        return QuerySet(cls)

    async def save(self) -> bool:
        """Save model instance to Milvus."""
        # Check if collection exists, create if not
        client = await ensure_connection()
        if not await client.has_collection(collection_name=self.collection_name):
            await self.create_collection()

        # Convert to dict and insert
        data = self.to_dict()
        result = await client.insert(collection_name=self.collection_name, data=[data])

        # Update primary key if auto_id is enabled
        primary_key = self._primary_key_field
        if not hasattr(self, primary_key) or getattr(self, primary_key) is None:
            if result.get("insert_count", 0) > 0:
                setattr(self, primary_key, result.get("primary_keys", [None])[0])

        return result.get("insert_count", 0) > 0

    async def delete(self) -> bool:
        """Delete model instance from Milvus."""
        client = await ensure_connection()
        primary_key = self._primary_key_field
        pk_value = getattr(self, primary_key, None)

        if pk_value is None:
            raise ValueError("Cannot delete instance without primary key")

        result = await client.delete(
            collection_name=self.collection_name, filter=f"{primary_key} == {pk_value}"
        )

        return result.get("delete_count", 0) > 0

    async def update(self, **kwargs) -> bool:
        """Update model instance with new values."""
        client = await ensure_connection()
        primary_key = self._primary_key_field
        pk_value = getattr(self, primary_key, None)

        if pk_value is None:
            raise ValueError("Cannot update instance without primary key")

        # Validate and update fields
        for field_name, value in kwargs.items():
            if field_name in self._fields:
                field = self._fields[field_name]
                if not field.validate(value):
                    raise ValueError(f"Invalid value for field '{field_name}': {value}")
                setattr(self, field_name, value)
            else:
                # Add to extra fields for dynamic schema
                if not hasattr(self, "_extra_fields"):
                    self._extra_fields = {}
                self._extra_fields[field_name] = value

        # Delete old instance and insert updated one
        # This is a workaround since Milvus doesn't support direct updates
        delete_result = await self.delete()
        if delete_result:
            # Reset primary key to None to let Milvus assign a new one
            # (since we're inserting a new record)
            setattr(self, primary_key, None)
            return await self.save()

        return False
