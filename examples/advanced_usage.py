"""
Milvus ORM 高级使用示例
"""

import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from milvus_orm import connections, Model
from milvus_orm.fields import IntField, CharField, VectorField, FloatField, BooleanField


# 定义复杂的文章模型
class Article(Model):
    id = IntField(primary_key=True, auto_id=True)
    title = CharField(max_length=200)
    content = CharField(max_length=5000)
    category = CharField(max_length=50)
    tags = CharField(max_length=500)  # 用逗号分隔的标签
    embedding = VectorField(dim=384)
    views = IntField(default=0)
    rating = FloatField(default=0.0)
    is_published = BooleanField(default=False)
    created_at = IntField()  # 时间戳
    
    class Meta:
        collection_name = "articles"
        connection_alias = "default"
    
    @classmethod
    def search_by_category(cls, category: str, limit: int = 10):
        """根据分类搜索"""
        return cls.objects.filter(category=category).limit(limit)
    
    @classmethod
    def get_popular_articles(cls, min_views: int = 100, limit: int = 10):
        """获取热门文章"""
        return cls.objects.filter(
            views__gt=min_views, 
            is_published=True
        ).order_by("-views").limit(limit)
    
    @classmethod
    def get_by_tags(cls, tags: list, limit: int = 10):
        """根据标签搜索（多个标签）"""
        queryset = cls.objects.all()
        
        for tag in tags:
            queryset = queryset.filter(tags__contains=tag)
        
        return queryset.limit(limit)


# 定义用户模型
class User(Model):
    id = IntField(primary_key=True, auto_id=True)
    username = CharField(max_length=50, unique=True)
    email = CharField(max_length=100)
    profile_embedding = VectorField(dim=128)
    is_active = BooleanField(default=True)
    created_at = IntField()
    
    class Meta:
        collection_name = "users"


def demo_advanced_queries():
    """演示高级查询功能"""
    print("=== 高级查询演示 ===")
    
    # 复杂过滤条件
    print("\n1. 复杂过滤查询")
    complex_results = Article.objects.filter(
        category="技术",
        views__gt=100,
        is_published=True
    ).order_by("-rating")
    print(f"技术类热门文章: {complex_results.count()} 条")
    
    # 组合查询
    print("\n2. 组合查询")
    combined_results = Article.objects.filter(
        title__contains="Python"
    ).exclude(
        category="娱乐"
    ).order_by("-created_at")
    print(f"Python相关非娱乐文章: {combined_results.count()} 条")
    
    # 自定义查询方法
    print("\n3. 自定义查询方法")
    tech_articles = Article.search_by_category("技术", limit=5)
    popular_articles = Article.get_popular_articles(min_views=50, limit=3)
    
    print(f"技术文章: {[a.title for a in tech_articles]}")
    print(f"热门文章: {[a.title for a in popular_articles]}")
    
    # 多标签搜索
    print("\n4. 多标签搜索")
    tagged_articles = Article.get_by_tags(["AI", "机器学习"], limit=5)
    print(f"AI和机器学习相关文章: {[a.title for a in tagged_articles]}")


def demo_batch_operations():
    """演示批量操作"""
    print("\n=== 批量操作演示 ===")
    
    # 批量创建
    articles_data = [
        {
            "title": f"批量文章{i}",
            "content": f"这是第{i}篇批量文章的内容...",
            "category": "批量测试",
            "tags": "批量,测试",
            "embedding": [float(i)/100] * 384,
            "views": i * 10,
            "rating": float(i) / 10,
            "is_published": True,
            "created_at": 1700000000 + i
        }
        for i in range(1, 6)
    ]
    
    created_articles = []
    for data in articles_data:
        article = Article.create(**data)
        created_articles.append(article)
        print(f"创建文章: {article.title}")
    
    # 批量更新
    print("\n批量更新视图数")
    for article in created_articles:
        article.views += 100
        article.save()
        print(f"更新文章 {article.title}: 视图数 -> {article.views}")


def demo_vector_search():
    """演示向量搜索功能"""
    print("\n=== 向量搜索演示 ===")
    
    # 创建一些带有向量的文章
    vector_articles = [
        {
            "title": "AI技术发展",
            "content": "人工智能技术的快速发展",
            "category": "技术",
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 381,
            "created_at": 1700000001
        },
        {
            "title": "机器学习应用",
            "content": "机器学习在实际应用中的案例",
            "category": "技术",
            "embedding": [0.15, 0.25, 0.35] + [0.0] * 381,
            "created_at": 1700000002
        },
        {
            "title": "旅游攻略",
            "content": "热门旅游目的地推荐",
            "category": "生活",
            "embedding": [0.8, 0.9, 0.7] + [0.0] * 381,
            "created_at": 1700000003
        }
    ]
    
    for data in vector_articles:
        Article.create(**data)
    
    # 向量搜索（需要Milvus索引支持）
    print("向量搜索演示")
    print("注意：向量搜索功能需要在Milvus中配置索引才能正常工作")
    
    # 这里展示向量搜索的代码结构
    search_vector = [0.1, 0.2, 0.3] + [0.0] * 381
    
    # 实际使用时需要取消注释并确保Milvus索引已创建
    # results = Article.objects.annotate(
    #     distance=Article.embedding.distance(search_vector)
    # ).filter(distance__lt=0.5).order_by("distance")
    
    # print(f"向量搜索结果: {[a.title for a in results]}")


def demo_error_handling():
    """演示错误处理"""
    print("\n=== 错误处理演示 ===")
    
    # 查询不存在的记录
    try:
        article = Article.objects.get(id=999999)
    except Article.DoesNotExist:
        print("✓ 捕获DoesNotExist异常: 记录不存在")
    
    # 可能返回多条记录的查询
    try:
        # 创建多个符合条件的记录
        Article.create(title="测试文章1", category="测试", created_at=1700000000)
        Article.create(title="测试文章2", category="测试", created_at=1700000000)
        
        article = Article.objects.filter(category="测试").get()
    except Article.MultipleObjectsReturned:
        print("✓ 捕获MultipleObjectsReturned异常: 找到多条记录")
    
    # 字段验证错误
    try:
        Article.create(title="A" * 300)  # 超过最大长度
    except ValueError as e:
        print(f"✓ 捕获字段验证错误: {e}")


def main():
    """主函数"""
    # 连接Milvus
    print("=== 连接Milvus ===")
    try:
        connections.create_connection(
            alias="default", 
            host='localhost',
            port='19530'
        )
        print("连接成功!")
    except Exception as e:
        print(f"连接失败: {e}")
        print("将继续演示，但部分操作可能无法执行")
    
    # 运行演示
    demo_advanced_queries()
    demo_batch_operations()
    demo_vector_search()
    demo_error_handling()
    
    # 清理
    print("\n=== 清理 ===")
    connections.close_all()
    print("连接已关闭")


if __name__ == "__main__":
    main()