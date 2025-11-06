# Milvus ORM

一个基于Milvus的Python ORM库，提供类似Django ORM的语法来管理和查询向量数据。

## 特性

- 🚀 **Django ORM风格**: 类似Django的API设计
- 📊 **向量支持**: 原生支持向量字段和向量搜索
- 🔍 **链式查询**: 支持链式API调用
- 🔗 **多连接管理**: 支持多个Milvus连接
- 🧩 **丰富字段**: 支持多种字段类型
- ✅ **类型提示**: 完善的类型注解支持

## 安装

### 从源码安装

```bash
cd milvus_orm
pip install -e .
```

### 依赖要求

- Python 3.7+
- pymilvus >= 2.0.0

## 快速开始

### 1. 定义模型

```python
from milvus_orm import Model
from milvus_orm.fields import IntField, CharField, VectorField

class Article(Model):
    id = IntField(primary_key=True)
    title = CharField(max_length=200)
    content = CharField(max_length=1000)
    embedding = VectorField(dim=384)
    
    class Meta:
        collection_name = "articles"
```

### 2. 连接数据库

```python
from milvus_orm import connections

# 连接Milvus
connections.create_connection(
    alias="default",
    host="localhost",  # 你的Milvus地址
    port="19530"
)
```

### 3. 基本操作

```python
# 创建记录
article = Article.create(
    id=1,
    title="Python ORM教程",
    content="这是一篇关于Python ORM的文章...",
    embedding=[0.1] * 384
)

# 查询记录
articles = Article.objects.filter(title__contains="Python")
article = Article.objects.get(id=1)

# 更新记录
article.title = "更新后的标题"
article.save()

# 删除记录
article.delete()
```

## 字段类型

### 基础字段

- `IntField` - 整数字段 (64位整数)
- `FloatField` - 浮点数字段  
- `CharField` - 字符串字段 (支持最大长度)
- `BooleanField` - 布尔字段

### 向量字段

- `VectorField` - 浮点向量字段 (必需指定维度dim)

### 字段选项

```python
# 主键字段
id = IntField(primary_key=True)

# 自动生成ID
id = IntField(primary_key=True, auto_id=True)

# 默认值
title = CharField(max_length=200, default="默认标题")

# 字段描述
description = CharField(max_length=500, description="文章描述")
```

## 查询API

### 基本查询

```python
# 所有记录
Article.objects.all()

# 条件查询
Article.objects.filter(title="Python")
Article.objects.exclude(title="Java")

# 排序
Article.objects.order_by("-id")  # 降序
Article.objects.order_by("title")  # 升序

# 限制和偏移
Article.objects.limit(10).offset(0)

# 计数
Article.objects.count()
```

### 比较操作符

```python
# 大于/小于
Article.objects.filter(id__gt=10)    # 大于
Article.objects.filter(id__lt=100)   # 小于
Article.objects.filter(id__gte=5)    # 大于等于
Article.objects.filter(id__lte=50)   # 小于等于

# 等于/不等于
Article.objects.filter(title__eq="Python")  # 等于
Article.objects.filter(title__ne="Java")   # 不等于
```

### 字符串操作

```python
# 包含查询
Article.objects.filter(title__contains="Python")
Article.objects.filter(title__startswith="AI")
Article.objects.filter(title__endswith="教程")

# IN查询
Article.objects.filter(id__in=[1, 2, 3])
```

### 向量搜索

```python
# 向量搜索 (需要创建索引)
search_vector = [0.1] * 384

# 基于距离的筛选
results = Article.objects.annotate(
    distance=Article.embedding.distance(search_vector)
).filter(distance__lt=0.5).order_by("distance")
```

## 模型选项

在模型的Meta类中可以设置以下选项：

```python
class Article(Model):
    id = IntField(primary_key=True)
    title = CharField(max_length=200)
    
    class Meta:
        collection_name = "articles"     # 集合名称
        connection_alias = "default"    # 连接别名
```

## 连接管理

### 多连接支持

```python
from milvus_orm import connections

# 创建多个连接
connections.create_connection("prod", host="production-host", port="19530")
connections.create_connection("dev", host="dev-host", port="19530")

# 使用特定连接
Article.objects(connection_alias="prod").all()
```

### 连接操作

```python
# 获取连接
conn = connections.get_connection("default")

# 关闭单个连接
connections.close_connection("default")

# 关闭所有连接
connections.close_all()
```

## 错误处理

```python
try:
    article = Article.objects.get(id=999)  # 不存在的ID
except Article.DoesNotExist:
    print("文章不存在")

try:
    article = Article.objects.filter(title="Python").get()  # 可能返回多条
except Article.MultipleObjectsReturned:
    print("找到多条记录，请使用更精确的查询条件")
```

## 高级用法

### 批量操作

```python
# 批量创建
articles = []
for i in range(10):
    article = Article(
        id=i+1,
        title=f"文章{i+1}",
        content=f"内容{i+1}",
        embedding=[float(i)/10] * 384
    )
    articles.append(article)

# 逐个保存
for article in articles:
    article.save()
```

### 自定义查询方法

```python
class Article(Model):
    # ... 字段定义
    
    @classmethod
    def search_by_keyword(cls, keyword: str):
        """根据关键词搜索"""
        return cls.objects.filter(
            title__contains=keyword
        ).order_by("-id")
    
    @classmethod
    def get_recent_articles(cls, limit: int = 10):
        """获取最近的文章"""
        return cls.objects.order_by("-id").limit(limit)
```

## 开发指南

### 运行测试

```bash
cd milvus_orm
pytest tests/
```

### 项目结构

```
milvus_orm/
├── src/
│   └── milvus_orm/
│       ├── __init__.py     # 包导出
│       ├── connections.py  # 连接管理
│       ├── fields.py       # 字段类型
│       ├── models.py        # 模型基类
│       └── query.py        # 查询集
├── examples/               # 使用示例
├── tests/                  # 测试文件
├── pyproject.toml         # 项目配置
└── README.md             # 项目文档
```

## 示例代码

查看 `examples/` 目录中的完整示例：

- `basic_usage.py` - 基础用法示例

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！