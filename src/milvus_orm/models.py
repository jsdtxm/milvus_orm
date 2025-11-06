"""
核心模型类，实现类似Django ORM的Model基类
"""

import inspect
from typing import Dict, Any, List, Optional, Type, TypeVar
from .fields import Field
from .query import QuerySet
from .connections import connections

T = TypeVar('T', bound='Model')


class ModelMeta(type):
    """模型元类，用于处理模型字段和元选项"""
    
    def __new__(cls, name, bases, attrs):
        # 跳过Model基类本身的处理
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        
        # 处理Meta类
        meta = attrs.pop('Meta', None)
        
        # 收集字段
        fields = {}
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field):
                fields[attr_name] = attr_value
                attr_value.name = attr_name
        
        # 设置字段到类属性
        attrs['_fields'] = fields
        
        # 处理元选项
        attrs['_meta'] = cls._process_meta(meta, name)
        
        # 创建新类
        new_class = super().__new__(cls, name, bases, attrs)
        
        return new_class
    
    @staticmethod
    def _process_meta(meta, class_name):
        """处理Meta类选项"""
        meta_attrs = {}
        
        if meta is not None:
            for attr_name in dir(meta):
                if not attr_name.startswith('_'):
                    meta_attrs[attr_name] = getattr(meta, attr_name)
        
        # 设置默认值
        meta_attrs.setdefault('collection_name', f"{class_name.lower()}")
        meta_attrs.setdefault('connection_alias', 'default')
        
        return type('Meta', (), meta_attrs)()


class Model(metaclass=ModelMeta):
    """模型基类"""
    
    class DoesNotExist(Exception):
        """记录不存在异常"""
        pass
    
    class MultipleObjectsReturned(Exception):
        """多个记录返回异常"""
        pass
    
    def __init__(self, **kwargs):
        """初始化模型实例"""
        # 设置字段默认值
        for field_name, field in self._fields.items():
            setattr(self, field_name, field.default)
        
        # 处理传入参数
        for key, value in kwargs.items():
            if key in self._fields:
                field = self._fields[key]
                setattr(self, key, field.validate(value))
            else:
                # 忽略未知字段
                pass
        
        self._saved = False
        self._changed_fields = set()
    
    @classmethod
    def _get_collection(cls, connection_alias: str = None):
        """获取或创建集合"""
        if connection_alias is None:
            connection_alias = cls._meta.connection_alias
        
        from pymilvus import Collection
        
        # 检查集合是否存在
        try:
            collection = Collection(cls._meta.collection_name, using=connection_alias)
            return collection
        except Exception:
            # 集合不存在，需要创建
            return cls._create_collection(connection_alias)
    
    @classmethod
    def _create_collection(cls, connection_alias: str):
        """创建集合"""
        from pymilvus import (
            CollectionSchema, FieldSchema, DataType,
            Collection, connections
        )
        
        # 构建字段定义
        fields = []
        for field_name, field in cls._fields.items():
            field_schema = field.to_milvus_schema()
            
            # 转换为Milvus FieldSchema
            if field.field_type.value == "FLOAT_VECTOR":
                schema = FieldSchema(
                    name=field_schema["name"],
                    dtype=DataType.FLOAT_VECTOR,
                    dim=field_schema["dim"]
                )
            elif field.field_type.value == "VARCHAR":
                schema = FieldSchema(
                    name=field_schema["name"],
                    dtype=DataType.VARCHAR,
                    max_length=field_schema["max_length"]
                )
            else:
                # 其他基本类型
                dtype_map = {
                    "INT8": DataType.INT8,
                    "INT16": DataType.INT16,
                    "INT32": DataType.INT32,
                    "INT64": DataType.INT64,
                    "FLOAT": DataType.FLOAT,
                    "DOUBLE": DataType.DOUBLE,
                    "BOOL": DataType.BOOL
                }
                schema = FieldSchema(
                    name=field_schema["name"],
                    dtype=dtype_map[field.field_type.value]
                )
            
            fields.append(schema)
        
        # 创建集合架构
        schema = CollectionSchema(fields, description=f"Collection for {cls.__name__}")
        
        # 创建集合
        collection = Collection(
            name=cls._meta.collection_name,
            schema=schema,
            using=connection_alias
        )
        
        # 为向量字段创建索引
        for field_name, field in cls._fields.items():
            if field.field_type.value == "FLOAT_VECTOR":
                index_params = {
                    "metric_type": field.metric_type,
                    "index_type": field.index_type,
                    "params": field.index_params
                }
                collection.create_index(field_name, index_params)
        
        return collection
    
    def save(self, connection_alias: str = None):
        """保存实例到数据库"""
        if connection_alias is None:
            connection_alias = self._meta.connection_alias
        
        collection = self._get_collection(connection_alias)
        
        # 构建插入数据
        data = {}
        for field_name, field in self._fields.items():
            value = getattr(self, field_name)
            if value is not None:
                data[field_name] = [value]
        
        # 插入数据
        try:
            result = collection.insert(data)
            
            # 如果是新记录且主键是自动生成的，设置主键值
            if not self._saved:
                primary_key_field = None
                for field_name, field in self._fields.items():
                    if field.primary_key and field.auto_id:
                        primary_key_field = field_name
                        break
                
                if primary_key_field and result.primary_keys:
                    setattr(self, primary_key_field, result.primary_keys[0])
            
            self._saved = True
            self._changed_fields.clear()
            
        except Exception as e:
            raise ValueError(f"Save failed: {e}")
    
    def delete(self, connection_alias: str = None):
        """删除实例"""
        if not self._saved:
            raise ValueError("Cannot delete unsaved instance")
        
        if connection_alias is None:
            connection_alias = self._meta.connection_alias
        
        collection = self._get_collection(connection_alias)
        
        # 构建删除表达式
        primary_key_field = None
        primary_key_value = None
        
        for field_name, field in self._fields.items():
            if field.primary_key:
                primary_key_field = field_name
                primary_key_value = getattr(self, field_name)
                break
        
        if not primary_key_field:
            raise ValueError("No primary key field defined")
        
        expr = f"{primary_key_field} = {primary_key_value}"
        
        try:
            collection.delete(expr)
            self._saved = False
        except Exception as e:
            raise ValueError(f"Delete failed: {e}")
    
    def __setattr__(self, name, value):
        """设置属性时跟踪变更"""
        if hasattr(self, '_fields') and name in self._fields:
            field = self._fields[name]
            validated_value = field.validate(value)
            
            # 检查值是否改变
            current_value = getattr(self, name, None)
            if current_value != validated_value:
                self._changed_fields.add(name)
            
            super().__setattr__(name, validated_value)
        else:
            super().__setattr__(name, value)
    
    def __str__(self):
        """字符串表示"""
        primary_key = None
        for field_name, field in self._fields.items():
            if field.primary_key:
                primary_key = getattr(self, field_name)
                break
        
        if primary_key is not None:
            return f"<{self.__class__.__name__}: {primary_key}>"
        else:
            return f"<{self.__class__.__name__}>"
    
    def __repr__(self):
        return str(self)
    
    # 类方法
    @classmethod
    def objects(cls, connection_alias: str = None) -> QuerySet:
        """获取查询集"""
        if connection_alias is None:
            connection_alias = cls._meta.connection_alias
        return QuerySet(cls, connection_alias)
    
    @classmethod
    def create(cls, **kwargs) -> T:
        """创建并保存新实例"""
        instance = cls(**kwargs)
        instance.save()
        return instance
    
    @classmethod
    def get_or_create(cls, defaults=None, **kwargs) -> (T, bool):
        """获取或创建实例"""
        if defaults is None:
            defaults = {}
        
        try:
            instance = cls.objects.get(**kwargs)
            return instance, False
        except cls.DoesNotExist:
            # 创建新实例
            create_kwargs = kwargs.copy()
            create_kwargs.update(defaults)
            instance = cls.create(**create_kwargs)
            return instance, True