"""
查询集实现，支持链式查询操作
"""

from typing import List, Dict, Any, Optional, Union, Iterator
from .fields import VectorField


class QuerySet:
    """查询集类，支持链式查询"""
    
    def __init__(self, model_class, connection_alias: str = "default"):
        self.model_class = model_class
        self.connection_alias = connection_alias
        self._filters: List[str] = []
        self._order_by: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._output_fields: List[str] = []
        self._expr: Optional[str] = None
        self._params: Dict[str, Any] = {}
    
    def filter(self, **kwargs) -> "QuerySet":
        """添加过滤条件"""
        qs = self._clone()
        
        for field_name, value in kwargs.items():
            if "__" in field_name:
                field, op = field_name.split("__", 1)
                if op == "contains":
                    qs._filters.append(f"{field} like '%{value}%'")
                elif op == "startswith":
                    qs._filters.append(f"{field} like '{value}%'")
                elif op == "endswith":
                    qs._filters.append(f"{field} like '%{value}'")
                elif op == "in":
                    if isinstance(value, (list, tuple)):
                        in_values = ",".join([f"'{v}'" for v in value])
                        qs._filters.append(f"{field} in ({in_values})")
                elif op in ["gt", "gte", "lt", "lte", "eq", "ne"]:
                    operator_map = {
                        "gt": ">", "gte": ">=", "lt": "<", 
                        "lte": "<=", "eq": "=", "ne": "!="
                    }
                    qs._filters.append(f"{field} {operator_map[op]} {value}")
            else:
                qs._filters.append(f"{field_name} = '{value}'")
        
        return qs
    
    def exclude(self, **kwargs) -> "QuerySet":
        """排除条件"""
        qs = self._clone()
        
        for field_name, value in kwargs.items():
            if "__" in field_name:
                field, op = field_name.split("__", 1)
                if op == "contains":
                    qs._filters.append(f"{field} not like '%{value}%'")
                elif op == "startswith":
                    qs._filters.append(f"{field} not like '{value}%'")
                elif op == "endswith":
                    qs._filters.append(f"{field} not like '%{value}'")
            else:
                qs._filters.append(f"{field_name} != '{value}'")
        
        return qs
    
    def order_by(self, *fields: str) -> "QuerySet":
        """排序"""
        qs = self._clone()
        qs._order_by = []
        
        for field in fields:
            if field.startswith("-"):
                qs._order_by.append(f"{field[1:]} desc")
            else:
                qs._order_by.append(f"{field} asc")
        
        return qs
    
    def limit(self, limit: int) -> "QuerySet":
        """限制结果数量"""
        qs = self._clone()
        qs._limit = limit
        return qs
    
    def offset(self, offset: int) -> "QuerySet":
        """偏移量"""
        qs = self._clone()
        qs._offset = offset
        return qs
    
    def only(self, *fields: str) -> "QuerySet":
        """只返回指定字段"""
        qs = self._clone()
        qs._output_fields = list(fields)
        return qs
    
    def annotate(self, **kwargs) -> "QuerySet":
        """添加注解字段（如向量距离）"""
        qs = self._clone()
        
        for field_name, expr in kwargs.items():
            if hasattr(expr, 'field') and hasattr(expr, 'vector'):
                # 向量距离注解
                vector_field = expr.field
                search_vector = expr.vector
                
                # 这里简化处理，实际应该构建更复杂的向量搜索表达式
                qs._params[f"search_vector_{field_name}"] = search_vector
                qs._expr = f"{vector_field.name} <-> $search_vector_{field_name}"
        
        return qs
    
    def all(self) -> "QuerySet":
        """返回所有记录"""
        return self._clone()
    
    def count(self) -> int:
        """返回记录数量"""
        from .connections import connections
        
        collection = self.model_class._get_collection(self.connection_alias)
        
        # 构建查询表达式
        expr = " and ".join(self._filters) if self._filters else ""
        
        try:
            result = collection.query(expr=expr, output_fields=["count(*)"])
            return len(result)
        except Exception:
            # 如果查询失败，返回0
            return 0
    
    def first(self):
        """返回第一条记录"""
        results = self.limit(1).execute()
        return results[0] if results else None
    
    def get(self, **kwargs):
        """获取单条记录"""
        qs = self.filter(**kwargs)
        results = qs.limit(2).execute()
        
        if len(results) == 0:
            raise self.model_class.DoesNotExist(f"{self.model_class.__name__} matching query does not exist")
        elif len(results) > 1:
            raise self.model_class.MultipleObjectsReturned(f"Multiple {self.model_class.__name__} matching query")
        
        return results[0]
    
    def exists(self) -> bool:
        """检查是否存在记录"""
        return self.count() > 0
    
    def execute(self) -> List:
        """执行查询并返回结果"""
        from .connections import connections
        
        collection = self.model_class._get_collection(self.connection_alias)
        
        # 构建查询参数
        params = {}
        
        # 构建表达式
        expr_parts = []
        if self._filters:
            expr_parts.append(" and ".join(self._filters))
        if self._expr:
            expr_parts.append(self._expr)
        
        expr = " and ".join(expr_parts) if expr_parts else ""
        
        # 设置输出字段
        output_fields = self._output_fields if self._output_fields else None
        
        # 执行查询
        try:
            result = collection.query(
                expr=expr or "",
                output_fields=output_fields,
                limit=self._limit,
                offset=self._offset
            )
            
            # 转换为模型实例
            instances = []
            for item in result:
                instance = self.model_class()
                for field_name, value in item.items():
                    setattr(instance, field_name, value)
                instances.append(instance)
            
            return instances
            
        except Exception as e:
            raise ValueError(f"Query failed: {e}")
    
    def __iter__(self) -> Iterator:
        """迭代器"""
        return iter(self.execute())
    
    def __getitem__(self, key):
        """切片操作"""
        if isinstance(key, slice):
            qs = self._clone()
            if key.start is not None:
                qs._offset = key.start
            if key.stop is not None:
                qs._limit = key.stop - (key.start or 0)
            return qs.execute()
        else:
            raise TypeError("QuerySet indices must be slices")
    
    def _clone(self) -> "QuerySet":
        """创建查询集副本"""
        qs = QuerySet(self.model_class, self.connection_alias)
        qs._filters = self._filters.copy()
        qs._order_by = self._order_by.copy()
        qs._limit = self._limit
        qs._offset = self._offset
        qs._output_fields = self._output_fields.copy()
        qs._expr = self._expr
        qs._params = self._params.copy()
        return qs