"""
Milvus ORM - A Django-like ORM wrapper for Milvus vector database.
"""

__version__ = "0.1.0"
__author__ = "Developer"
__email__ = "dev@example.com"

from .connections import connections
from .models import Model
from .query import QuerySet

__all__ = ["Model", "QuerySet", "connections"]