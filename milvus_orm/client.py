"""
Client module for milvus_orm. Manages connection to Milvus using AsyncMilvusClient.
"""

from typing import Optional, Dict, Any
from pymilvus import AsyncMilvusClient


# Global client instance
_client: Optional[AsyncMilvusClient] = None
_client_config: Optional[Dict[str, Any]] = None


async def connect(
    uri: str = "http://localhost:19530",
    token: Optional[str] = None,
    **kwargs
) -> AsyncMilvusClient:
    """
    Connect to Milvus server.
    
    Args:
        uri: Milvus server URI
        token: Authentication token
        **kwargs: Additional connection parameters
    
    Returns:
        AsyncMilvusClient instance
    """
    global _client, _client_config
    
    # If client is already connected, return it
    if _client is not None:
        return _client
    
    # Create new client
    _client_config = {
        "uri": uri,
        "token": token,
        **kwargs
    }
    
    _client = AsyncMilvusClient(**_client_config)
    return _client


async def disconnect() -> None:
    """
    Disconnect from Milvus server.
    """
    global _client, _client_config
    
    if _client is not None:
        await _client.close()
        _client = None
        _client_config = None


async def get_client() -> Optional[AsyncMilvusClient]:
    """
    Get the current Milvus client instance.
    
    Returns:
        AsyncMilvusClient instance if connected, None otherwise
    """
    global _client
    
    # If client is not connected, connect with default settings
    if _client is None:
        await connect()
    
    return _client


async def ensure_connection() -> AsyncMilvusClient:
    """
    Ensure that we have an active connection to Milvus.
    
    Returns:
        AsyncMilvusClient instance
    
    Raises:
        ConnectionError: If connection cannot be established
    """
    client = await get_client()
    if client is None:
        raise ConnectionError("Failed to connect to Milvus")
    return client