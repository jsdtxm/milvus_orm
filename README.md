# milvus_orm

一个基于Milvus向量数据库的异步ORM库，提供类似Django ORM的语法来管理和查询数据。

## 特性

- 提供类似Django ORM的简洁API
- 完全异步支持，使用AsyncMilvusClient
- 支持多种数据类型：FLOAT_VECTOR、VARCHAR、JSON、INT64、FLOAT等
- 支持向量搜索和标量过滤
- 支持动态字段

## 安装

```bash
pip install pymilvus
# 然后直接使用本库
```

## 快速开始

### 1. 连接到Milvus

```python
import asyncio
from milvus_orm import connect, disconnect, Model, INT64, VARCHAR, FLOAT_VECTOR, JSON

async def main():
    # 连接到Milvus服务器
    await connect(uri="http://localhost:19530")
    
    # 使用完后断开连接
    await disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 定义模型

```python
class Product(Model):
    # 设置集合名称（可选，默认为类名小写）
    collection_name = "products"
    
    # 定义字段
    id = INT64(primary_key=True)
    name = VARCHAR(max_length=255)
    description = VARCHAR(max_length=1000)
    price = FLOAT()
    vector = FLOAT_VECTOR(dim=768)  # 768维向量
    metadata = JSON(nullable=True)  # 可选的JSON字段
```

### 3. 创建集合

```python
# 创建集合（如果不存在）
await Product.create_collection()
```

### 4. 插入数据

```python
# 单个插入
product = Product(
    id=1,
    name="智能手表",
    description="多功能智能手表",
    price=999.99,
    vector=[0.1, 0.2, ..., 0.9],  # 768维向量
    metadata={"brand": "品牌A", "color": "黑色"}
)
await product.save()

# 批量插入
products = [
    Product(
        id=i,
        name=f"产品{i}",
        vector=[0.1*i, 0.2*i, ..., 0.9*i],
        price=100.0*i
    )
    for i in range(2, 10)
]
count = await Product.bulk_create(products)
print(f"成功创建了 {count} 个产品")
```

### 5. 查询数据

```python
# 获取单个对象
product = await Product.objects().get(id=1)

# 过滤查询
products = await Product.objects().filter("price > 500").all()

# 限制返回数量
products = await Product.objects().limit(10).all()

# 分页
products = await Product.objects().offset(10).limit(10).all()

# 计数
count = await Product.objects().filter("price > 500").count()

# 向量搜索
query_vector = [0.1, 0.2, ..., 0.9]  # 768维查询向量
results = await Product.objects().search(
    vector=query_vector,
    field_name="vector",
    metric_type="L2",  # 距离度量方式
    limit=5
).all()
```

### 6. 更新和删除

```python
# 更新对象
product = await Product.objects().get(id=1)
await product.update(price=899.99, name="智能手表Pro")

# 删除对象
await product.delete()

# 批量删除
count = await Product.objects().filter("price < 200").delete()
print(f"删除了 {count} 个产品")
```

## 字段类型

目前支持的字段类型：

- `INT64` - 64位整数
- `VARCHAR` - 可变长度字符串
- `FLOAT` - 浮点数
- `JSON` - JSON类型，用于存储结构化数据
- `FLOAT_VECTOR` - 密集浮点向量
- `SPARSE_FLOAT_VECTOR` - 稀疏浮点向量

## 查询操作符

Milvus支持的过滤操作符：

- `==` - 等于
- `!=` - 不等于
- `>` - 大于
- `<` - 小于
- `>=` - 大于等于
- `<=` - 小于等于
- `&&` - 逻辑与
- `||` - 逻辑或
- `()` - 括号优先级

## 注意事项

- 本库使用异步API，需要在异步环境中使用
- 对于更新操作，由于Milvus限制，采用了先删除后插入的方式
- 确保在使用前已经正确安装并启动了Milvus服务

## 开发环境

- Python 3.7+
- pymilvus

## 许可证

MIT