"""
Query module for milvus_orm. Defines the QuerySet class for async query operations.
"""

from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Type, TypeVar

from milvus_orm.exceptions import DoesNotExist, MultipleObjectsReturned

from .client import ensure_connection
from .fields import SparseFloatVectorField

if TYPE_CHECKING:
    from .models import Model

M = TypeVar("M", bound="Model")


class QuerySet(Generic[M]):
    """Async query set for Milvus models."""

    def __init__(self, model_class: Type[M]):
        self.model_class = model_class
        self._filter: Optional[str] = None
        self._limit: int = 1000
        self._offset: int = 0
        self._output_fields: Optional[List[str]] = None
        self._search_params: Optional[Dict[str, Any]] = None
        self._vector_field: Optional[str] = None

        self._model_fields: Optional[List[str]] = None
        self._defer_fields: List[str] = []

        self._consistency_level: Optional[str] = None
        self._collection_name: Optional[str] = None

    def get_collection_name(self) -> str:
        """Get the collection name to query."""
        if self.model_class.Meta.dynamic and not self._collection_name:
            raise ValueError("Dynamic collection must specify collection_name")
        return (
            self._collection_name
            if self._collection_name
            else self.model_class.Meta.collection_name
        )

    async def create(self, **kwargs) -> M:
        """Create a new instance of the model."""
        instance = self.model_class(
            _collection_name=self.get_collection_name(), **kwargs
        )
        await instance.save()
        return instance

    def on(self, collection_name: str) -> "QuerySet[M]":
        """Set the collection name to query."""
        qs = self._clone()
        qs._collection_name = collection_name
        return qs

    def filter(self, expr: str) -> "QuerySet[M]":
        """Add filter expression to the query."""
        qs = self._clone()
        qs._filter = expr
        return qs

    def limit(self, limit: int) -> "QuerySet[M]":
        """Set maximum number of results to return."""
        qs = self._clone()
        qs._limit = limit
        return qs

    def offset(self, offset: int) -> "QuerySet[M]":
        """Set offset for pagination."""
        qs = self._clone()
        qs._offset = offset
        return qs

    def only(self, *fields: str) -> "QuerySet[M]":
        """Specify fields to return."""
        qs = self._clone()
        qs._output_fields = list(fields)
        return qs

    def defer(self, *fields: str) -> "QuerySet[M]":
        """Specify fields to defer loading."""
        qs = self._clone()
        qs._defer_fields = list(fields)
        return qs

    def search(self, vector: List[float], field_name: str, **kwargs) -> "QuerySet[M]":
        """Configure vector search parameters."""
        qs = self._clone()
        qs._vector_field = field_name
        qs._search_params = {
            "vector": vector,
            "field_name": field_name,
            "filter": qs._filter,
            **kwargs,
        }
        return qs

    def _clone(self) -> "QuerySet[M]":
        """Clone the query set."""
        qs = QuerySet(self.model_class)
        qs._filter = self._filter
        qs._limit = self._limit
        qs._offset = self._offset
        qs._output_fields = self._output_fields
        qs._defer_fields = self._defer_fields
        qs._search_params = self._search_params.copy() if self._search_params else None
        qs._vector_field = self._vector_field
        qs._consistency_level = self._consistency_level
        qs._collection_name = self._collection_name
        return qs

    def _get_model_fields(self):
        if self._model_fields:
            return self._model_fields
        self._model_fields = [
            k
            for k, v in self.model_class._fields.items()
            if not isinstance(v, SparseFloatVectorField)
            and k not in set(self._defer_fields)
        ]
        return self._model_fields

    async def get(self, **kwargs) -> M:
        """Get a single instance matching the filter."""
        # Build filter expression from kwargs
        if kwargs:
            expr_parts = []
            for key, value in kwargs.items():
                if isinstance(value, str):
                    expr_parts.append(f"{key} == '{value}'")
                else:
                    expr_parts.append(f"{key} == {value}")
            expr = " && ".join(expr_parts)

            qs = self.filter(expr).limit(2)
        else:
            qs = self.limit(2)

        results = await qs.all()

        if len(results) == 0:
            raise DoesNotExist(
                f"{self.model_class.__name__} matching query does not exist."
            )
        if len(results) > 1:
            raise MultipleObjectsReturned(
                f"get() returned more than one {self.model_class.__name__} -- it returned {len(results)!r}!"
            )

        return results[0]

    async def all(self) -> List[M]:
        """Return all instances matching the query."""
        client = await ensure_connection()

        # Check if collection exists
        if not await client.has_collection(collection_name=self.get_collection_name()):
            return []

        # Load collection if not loaded
        if not await client.has_collection(
            collection_name=self.get_collection_name(), check_loaded=True
        ):
            await client.load_collection(collection_name=self.get_collection_name())

        # Determine which method to use: search or query
        if self._search_params and self._vector_field:
            # Use vector search
            results = await client.search(
                collection_name=self.get_collection_name(),
                data=[self._search_params["vector"]],
                filter=self._filter,
                anns_field=self._search_params["field_name"],
                limit=self._limit,
                output_fields=self._output_fields or self._get_model_fields(),
                consistency_level=self._consistency_level
                or self.model_class.Meta.consistency_level,
                **{
                    k: v
                    for k, v in self._search_params.items()
                    if k not in ["vector", "field_name", "limit", "offset", "filter"]
                },
            )

            # Convert search results to model instances
            instances = []
            for result in results[0]:  # results is a list of result lists
                entity = result.entity
                data = entity.to_dict()
                instances.append(self.model_class(_from_result=True, **data["entity"]))

            # Apply offset
            if self._offset > 0:
                instances = instances[self._offset :]

            return instances
        else:
            # Use scalar query
            results = await client.query(
                collection_name=self.get_collection_name(),
                filter=self._filter or "",
                limit=self._limit,
                offset=self._offset,
                output_fields=self._output_fields or self._get_model_fields(),
                consistency_level=self._consistency_level
                or self.model_class.Meta.consistency_level,
            )

            # Convert query results to model instances
            return [self.model_class(_from_result=True, **item) for item in results]

    async def count(self) -> int:
        """Count instances matching the query."""
        client = await ensure_connection()

        # Check if collection exists
        if not await client.has_collection(collection_name=self.get_collection_name()):
            return 0

        # Load collection if not loaded
        if not await client.has_collection(
            collection_name=self.get_collection_name(), check_loaded=True
        ):
            await client.load_collection(collection_name=self.get_collection_name())

        # Use query with limit=0 to get count
        results = await client.query(
            collection_name=self.get_collection_name(),
            filter=self._filter or "",
            consistency_level=self._consistency_level
            or self.model_class.Meta.consistency_level,
            # limit=0,
            output_fields=["count(*)"],
        )

        return results[0]["count(*)"]

    async def delete(self) -> int:
        """Delete all instances matching the query."""
        client = await ensure_connection()

        # Check if collection exists
        if not await client.has_collection(collection_name=self.get_collection_name()):
            return 0

        # Delete using filter
        result = await client.delete(
            collection_name=self.get_collection_name(),
            filter=self._filter or "",
        )

        return result.get("delete_count", 0)

    async def first(self) -> Optional[M]:
        """Return the first instance matching the query."""
        results = await self.limit(1).all()
        return results[0] if results else None

    async def last(self) -> Optional[M]:
        """Return the last instance matching the query."""
        # Milvus doesn't support ordering directly, so we get all and take last
        results = await self.all()
        return results[-1] if results else None
