"""
Milvus ORM 基础测试
"""

import sys
import os
import pytest

# 添加项目路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from milvus_orm import connections, Model
from milvus_orm.fields import IntField, CharField, VectorField


# 测试模型
class TestModel(Model):
    id = IntField(primary_key=True)
    name = CharField(max_length=50)
    description = CharField(max_length=200)
    vector = VectorField(dim=128)
    
    class Meta:
        collection_name = "test_collection"


class TestMilvusORM:
    """Milvus ORM 测试类"""
    
    def setup_class(self):
        """测试前设置"""
        # 创建测试连接 (使用mock连接，避免依赖真实Milvus)
        try:
            connections.create_connection(
                alias="test",
                host="localhost",
                port="19530"
            )
        except Exception:
            # 如果无法连接真实Milvus，跳过真实测试
            pytest.skip("无法连接Milvus，跳过真实测试")
    
    def teardown_class(self):
        """测试后清理"""
        connections.close_all()
    
    def test_model_creation(self):
        """测试模型创建"""
        # 测试模型字段定义
        assert hasattr(TestModel, '_fields')
        assert 'id' in TestModel._fields
        assert 'name' in TestModel._fields
        assert 'vector' in TestModel._fields
        
        # 测试字段类型
        assert TestModel._fields['id'].primary_key == True
        assert TestModel._fields['name'].max_length == 50
        assert TestModel._fields['vector'].dim == 128
    
    def test_instance_creation(self):
        """测试实例创建"""
        instance = TestModel(
            id=1,
            name="测试实例",
            description="这是一个测试实例",
            vector=[0.1] * 128
        )
        
        assert instance.id == 1
        assert instance.name == "测试实例"
        assert len(instance.vector) == 128
    
    def test_field_validation(self):
        """测试字段验证"""
        # 测试整数验证
        instance = TestModel(id="123")  # 字符串应该转换为整数
        assert instance.id == 123
        
        # 测试字符串长度验证
        with pytest.raises(ValueError):
            TestModel(name="a" * 100)  # 超过最大长度
    
    def test_queryset_methods(self):
        """测试查询集方法"""
        # 测试查询集创建
        qs = TestModel.objects()
        assert qs.model_class == TestModel
        
        # 测试链式调用
        qs = qs.filter(id__gt=0).order_by("-id").limit(10)
        assert isinstance(qs, type(qs))
    
    def test_meta_options(self):
        """测试元选项"""
        assert hasattr(TestModel, '_meta')
        assert TestModel._meta.collection_name == "test_collection"
        assert TestModel._meta.connection_alias == "default"


# 测试字段类型
class TestFields:
    """字段类型测试"""
    
    def test_int_field(self):
        """测试整数字段"""
        field = IntField(primary_key=True)
        assert field.primary_key == True
        assert field.field_type.value == "INT64"
        
        # 测试验证
        assert field.validate("123") == 123
        assert field.validate(456) == 456
    
    def test_char_field(self):
        """测试字符串字段"""
        field = CharField(max_length=50)
        assert field.max_length == 50
        assert field.field_type.value == "VARCHAR"
        
        # 测试验证
        assert field.validate("test") == "test"
        assert field.validate(123) == "123"
        
        # 测试长度限制
        with pytest.raises(ValueError):
            field.validate("a" * 100)
    
    def test_vector_field(self):
        """测试向量字段"""
        field = VectorField(dim=128)
        assert field.dim == 128
        assert field.field_type.value == "FLOAT_VECTOR"
        
        # 测试验证
        vector = [0.1] * 128
        assert field.validate(vector) == vector
        
        # 测试维度不匹配
        with pytest.raises(ValueError):
            field.validate([0.1] * 100)
    
    def test_vector_distance(self):
        """测试向量距离"""
        field = VectorField(dim=128)
        field.name = "embedding"
        vector = [0.1] * 128
        
        distance = field.distance(vector)
        assert distance.field == field
        assert distance.vector == vector
        
        # 测试距离表达式
        assert distance < 0.5 == "embedding < 0.5"
        assert distance > 0.1 == "embedding > 0.1"


# 测试连接管理
class TestConnections:
    """连接管理测试"""
    
    def test_connection_creation(self):
        """测试连接创建"""
        # 创建连接
        handler = connections.create_connection(
            alias="test_conn",
            host="test_host",
            port="12345"
        )
        
        assert handler.alias == "test_conn"
        assert "test_conn" in connections._connections
        
        # 清理
        connections.close_connection("test_conn")
    
    def test_multiple_connections(self):
        """测试多连接管理"""
        # 创建多个连接
        connections.create_connection("conn1", host="host1")
        connections.create_connection("conn2", host="host2")
        
        assert len(connections._connections) >= 2
        
        # 清理
        connections.close_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])