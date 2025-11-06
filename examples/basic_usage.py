"""
Milvus ORM 基础使用示例
"""

import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from milvus_orm import connections, Model
from milvus_orm.fields import IntField, CharField, VectorField


# 定义文章模型
class Article(Model):
    id = IntField(primary_key=True)
    title = CharField(max_length=200)
    content = CharField(max_length=1000)
    embedding = VectorField(dim=384)
    
    class Meta:
        collection_name = "articles"


# 定义用户模型
class User(Model):
    id = IntField(primary_key=True, auto_id=True)
    username = CharField(max_length=50)
    email = CharField(max_length=100)
    
    class Meta:
        collection_name = "users"


def main():
    # 1. 连接Milvus
    print("=== 连接Milvus ===")
    connections.create_connection(
        alias="default", 
        host='localhost',  # 修改为你的Milvus地址
        port='19530'
    )
    print("连接成功!")
    
    # 2. 创建记录
    print("\n=== 创建记录 ===")
    
    # 创建文章
    article1 = Article(
        id=1,
        title="Python ORM 教程",
        content="这是一篇关于Python ORM的文章...",
        embedding=[0.1] * 384  # 示例向量
    )
    article1.save()
    print(f"创建文章: {article1}")
    
    article2 = Article(
        id=2,
        title="Milvus 向量数据库",
        content="Milvus是一个开源的向量数据库...",
        embedding=[0.2] * 384
    )
    article2.save()
    print(f"创建文章: {article2}")
    
    # 使用快捷方式创建
    article3 = Article.create(
        id=3,
        title="机器学习应用",
        content="机器学习在现代应用中的使用...",
        embedding=[0.3] * 384
    )
    print(f"快捷创建文章: {article3}")
    
    # 3. 查询记录
    print("\n=== 查询记录 ===")
    
    # 获取所有记录
    all_articles = Article.objects.all()
    print(f"所有文章数量: {all_articles.count()}")
    
    # 条件查询
    python_articles = Article.objects.filter(title__contains="Python")
    print(f"包含'Python'的文章: {[str(a) for a in python_articles]}")
    
    # 获取单个记录
    try:
        article = Article.objects.get(id=1)
        print(f"获取ID=1的文章: {article}")
    except Article.DoesNotExist:
        print("文章不存在")
    
    # 4. 更新记录
    print("\n=== 更新记录 ===")
    article = Article.objects.get(id=1)
    article.title = "更新后的Python ORM教程"
    article.save()
    print(f"更新后的文章: {article}")
    
    # 5. 删除记录
    print("\n=== 删除记录 ===")
    article = Article.objects.get(id=3)
    article.delete()
    print(f"删除文章: {article}")
    print(f"剩余文章数量: {Article.objects.count()}")
    
    # 6. 高级查询示例
    print("\n=== 高级查询示例 ===")
    
    # 多条件查询
    results = Article.objects.filter(
        title__contains="教程", 
        id__gt=0
    ).order_by("-id")
    print(f"多条件查询结果: {[str(a) for a in results]}")
    
    # 限制和偏移
    limited_results = Article.objects.order_by("id").limit(1)
    print(f"限制查询结果: {[str(a) for a in limited_results]}")
    
    # 7. 向量搜索示例 (需要实际向量数据)
    print("\n=== 向量搜索示例 ===")
    # 这里展示向量搜索的基本用法，实际使用时需要替换为真实的向量数据
    search_vector = [0.1] * 384
    
    # 注意：向量搜索需要在Milvus中创建索引后才能正常工作
    # results_with_distance = Article.objects.annotate(
    #     distance=Article.embedding.distance(search_vector)
    # ).filter(distance__lt=0.5).order_by("distance")
    # print("向量搜索结果:", [str(a) for a in results_with_distance])
    
    print("向量搜索功能需在Milvus中配置索引后使用")
    
    # 8. 关闭连接
    print("\n=== 关闭连接 ===")
    connections.close_all()
    print("连接已关闭")


if __name__ == "__main__":
    main()