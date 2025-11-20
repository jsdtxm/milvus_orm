"""
Models module for milvus_orm. Defines the Model base class and related functionality.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

from pymilvus import CollectionSchema, FieldSchema
from pymilvus.milvus_client.index import IndexParams

from .client import ensure_connection
from .exceptions import NotContainsVectorField
from .fields import BigIntField, Field, VectorField
from .query import QuerySet
from .utils import classproperty

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

        contains_vector_field = False

        for key, value in attrs.items():
            if isinstance(value, Field):
                value.name = key
                fields[key] = value
                if value.primary_key:
                    primary_key_field = key
            if not contains_vector_field and isinstance(value, VectorField):
                contains_vector_field = True

        if not contains_vector_field:
            raise NotContainsVectorField(
                "Model must contain at least one vector field."
            )

        meta = attrs.get("Meta")
        meta_info_params = {"collection_name": name.lower()} | (
            {
                k: v
                for k, v in vars(meta).items()
                if not k.startswith("__") and not callable(v)
            }
            if meta
            else {}
        )

        attrs["Meta"] = MetaInfo(**meta_info_params)

        # If no primary key is defined, add a default one
        if not primary_key_field:
            id_field = BigIntField(primary_key=True, auto_id=True)
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


class MetaInfo:
    """
    Metadata class for Model.
    """

    def __init__(self, collection_name: str, enable_dynamic_field: bool = False):
        self.collection_name = collection_name
        self.enable_dynamic_field = enable_dynamic_field


class Model(object, metaclass=ModelMeta):
    """
    Base class for all milvus_orm models.
    Mapping to collection in Milvus.
    """

    # Will be set by metaclass
    _fields: Dict[str, Field] = {}
    _primary_key_field: str = "id"

    Meta: MetaInfo

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
        for field in cls._fields.values():
            field_schema = FieldSchema(**field.to_milvus_type())
            fields.append(field_schema)

        schema = CollectionSchema(
            fields=fields,
            auto_id=False,
            enable_dynamic_field=cls.Meta.enable_dynamic_field,
        )

        return schema

    @classmethod
    def _get_index_params(cls) -> IndexParams:
        """Generate index params for collection."""
        index_params = IndexParams()
        for field_name, field in cls._fields.items():
            if isinstance(field, VectorField):
                index_params.add_index(
                    field_name=field_name,
                    index_type=field.index_type,
                    # metric_type="L2",
                    # params={"nlist": 1024},
                )
        return index_params

    @classmethod
    async def create_collection(cls) -> bool:
        """Create collection in Milvus based on model schema."""
        client = await ensure_connection()
        schema = cls._get_schema()
        index_params = cls._get_index_params()

        # Check if collection already exists
        if await client.has_collection(collection_name=cls.Meta.collection_name):
            return False

        # Create the collection
        await client.create_collection(
            collection_name=cls.Meta.collection_name,
            schema=schema,
            index_params=index_params,
        )
        return True

    @classmethod
    async def drop_collection(cls) -> bool:
        """Drop collection from Milvus."""
        client = await ensure_connection()

        # Check if collection exists
        if not await client.has_collection(collection_name=cls.Meta.collection_name):
            return False

        # Drop the collection
        await client.drop_collection(collection_name=cls.Meta.collection_name)
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
        if not await client.has_collection(collection_name=cls.Meta.collection_name):
            await cls.create_collection()

        # Convert instances to dictionaries
        data = [instance.to_dict() for instance in instances]

        # Insert data in bulk
        print(data)
        result = await client.insert(
            collection_name=cls.Meta.collection_name, data=data
        )

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

    @classproperty
    def objects(cls: Type[M]) -> "QuerySet[M]":
        """Return a QuerySet for the model."""

        return QuerySet(cls)

    if TYPE_CHECKING:

        @classmethod
        @property
        def objects(cls) -> "QuerySet[M]": ...

    async def save(self) -> bool:
        """Save model instance to Milvus."""
        # Check if collection exists, create if not
        client = await ensure_connection()
        if not await client.has_collection(collection_name=self.Meta.collection_name):
            await self.create_collection()

        # Convert to dict and insert
        data = self.to_dict()
        result = await client.insert(
            collection_name=self.Meta.collection_name, data=[data]
        )

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
            collection_name=self.Meta.collection_name,
            filter=f"{primary_key} == {pk_value}",
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
