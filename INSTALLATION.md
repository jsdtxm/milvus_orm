# 安装和使用指南

## 系统要求

- Python 3.7 或更高版本
- Milvus 2.0 或更高版本
- pymilvus >= 2.0.0

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd milvus_orm
```

### 2. 安装依赖

使用 pip 安装项目：

```bash
# 安装开发版本（推荐）
pip install -e .

# 或者使用普通安装
pip install .
```

### 3. 验证安装

```python
import milvus_orm
print("Milvus ORM 安装成功!")
```

## 配置Milvus

### 1. 启动Milvus服务

首先确保你的Milvus服务正在运行：

```bash
# 使用Docker启动Milvus（推荐）
docker-compose up -d

# 或者使用官方安装方式
# 参考：https://milvus.io/docs/install_standalone-docker.md
```

### 2. 连接配置

在你的应用中配置Milvus连接：

```python
from milvus_orm import connections

# 建立连接
connections.create_connection(
    alias="default",
    host="localhost",  # 你的Milvus地址
    port="19530"        # Milvus服务端口
)
```

## 快速开始

### 1. 定义你的第一个模型

```python
from milvus_orm import Model
from milvus_orm.fields import IntField, CharField, VectorField

class Article(Model):
    id = IntField(primary_key=True)
    title = CharField(max_length=200)
    content = CharField(max_length=1000)
    embedding = VectorField(dim=384)  # 根据你的向量维度调整
    
    class Meta:
        collection_name = "articles"
```

### 2. 基本操作示例

```python
# 创建记录
article = Article.create(
    id=1,
    title="Python ORM教程",
    content="这是一篇关于Python ORM的文章...",
    embedding=[0.1] * 384  # 示例向量
)

# 查询记录
articles = Article.objects.filter(title__contains="Python")
print(f"找到 {len(articles)} 篇文章")

# 更新记录
article.title = "更新后的标题"
article.save()

# 删除记录
article.delete()
```

## 常见问题

### 1. 连接失败

**问题**: `ConnectionError: Failed to connect to Milvus`

**解决方案**:
- 检查Milvus服务是否正常运行
- 确认主机和端口配置正确
- 检查防火墙设置

```bash
# 检查Milvus状态
docker ps | grep milvus

# 测试端口连接
telnet localhost 19530
```

### 2. 向量维度不匹配

**问题**: `ValueError: Vector dimension mismatch`

**解决方案**:
- 确保VectorField的dim参数与实际向量维度一致
- 检查向量数据的维度

```python
# 正确的向量维度配置
embedding = VectorField(dim=384)  # 384维向量

# 正确的向量数据
vector = [0.1] * 384  # 384维向量
```

### 3. 集合不存在

**问题**: `Collection not found`

**解决方案**:
- 确保模型已正确保存过至少一条记录
- 或者手动创建集合

```python
# 确保调用save方法创建集合
article = Article(title="测试", embedding=[0.1]*384)
article.save()  # 这会自动创建集合
```

## 性能优化建议

### 1. 批量操作

```python
# 批量创建（推荐）
articles = []
for i in range(100):
    article = Article(title=f"文章{i}", embedding=[0.1]*384)
    articles.append(article)

# 逐个保存
for article in articles:
    article.save()
```

### 2. 索引优化

对于向量字段，确保创建合适的索引：

```python
# 在模型定义中配置向量索引
embedding = VectorField(
    dim=384,
    metric_type="L2",  # 距离度量方式
    index_type="IVF_FLAT",  # 索引类型
    index_params={"nlist": 128}  # 索引参数
)
```

### 3. 查询优化

- 使用适当的过滤条件减少返回数据量
- 只选择需要的字段
- 合理使用分页

```python
# 优化的查询
articles = Article.objects.filter(
    category="技术"
).only("title", "created_at").limit(10)
```

## 测试

运行测试确保一切正常：

```bash
# 运行基础测试
pytest tests/

# 运行特定测试文件
pytest tests/test_basic.py -v
```

## 下一步

- 查看 [examples/](examples/) 目录中的完整示例
- 阅读 [API文档](API.md) 了解详细接口
- 查看 [高级用法](ADVANCED.md) 了解复杂场景

## 获取帮助

如果遇到问题：

1. 查看本文档的常见问题部分
2. 检查Milvus官方文档
3. 在GitHub Issues中提问
4. 查看示例代码

## 版本兼容性

| Milvus ORM版本 | Milvus版本 | Python版本 |
|---------------|------------|------------|
| 1.0.0         | 2.0+       | 3.7+       |

---

**注意**: 本文档基于当前版本编写，具体功能可能随版本更新而变化。