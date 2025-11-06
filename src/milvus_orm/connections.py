"""
连接管理器，用于管理Milvus数据库连接
"""

from typing import Dict, Any, Optional
import pymilvus


class ConnectionHandler:
    """单个连接处理器"""
    
    def __init__(self, alias: str, **kwargs):
        self.alias = alias
        self.connection = None
        self.connection_params = kwargs
        
    def connect(self):
        """建立连接"""
        if self.connection is None:
            self.connection = pymilvus.connections.connect(
                alias=self.alias, **self.connection_params
            )
        return self.connection
    
    def close(self):
        """关闭连接"""
        if self.connection:
            pymilvus.connections.disconnect(self.alias)
            self.connection = None
    
    def get_connection(self):
        """获取连接，如果未连接则自动连接"""
        return self.connect()


class ConnectionManager:
    """连接管理器"""
    
    def __init__(self):
        self._connections: Dict[str, ConnectionHandler] = {}
        self._default_alias = 'default'
    
    def create_connection(self, alias: str = 'default', **kwargs):
        """创建新的连接"""
        if alias in self._connections:
            raise ValueError(f"Connection alias '{alias}' already exists")
        
        handler = ConnectionHandler(alias, **kwargs)
        self._connections[alias] = handler
        
        # 设置第一个连接为默认连接
        if len(self._connections) == 1:
            self._default_alias = alias
        
        return handler
    
    def get_connection(self, alias: str = None):
        """获取指定别名的连接"""
        if alias is None:
            alias = self._default_alias
        
        if alias not in self._connections:
            raise ValueError(f"Connection alias '{alias}' does not exist")
        
        return self._connections[alias].get_connection()
    
    def close_connection(self, alias: str = None):
        """关闭指定别名的连接"""
        if alias is None:
            alias = self._default_alias
        
        if alias in self._connections:
            self._connections[alias].close()
    
    def close_all(self):
        """关闭所有连接"""
        for handler in self._connections.values():
            handler.close()
        self._connections.clear()
    
    def set_default(self, alias: str):
        """设置默认连接别名"""
        if alias not in self._connections:
            raise ValueError(f"Connection alias '{alias}' does not exist")
        self._default_alias = alias


# 全局连接管理器实例
connections = ConnectionManager()