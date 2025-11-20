"""
示例代码：展示milvus_orm的基本使用方法
"""

import asyncio

from milvus_orm import Model, connect, disconnect
from milvus_orm.exceptions import DoesNotExist
from milvus_orm.fields import (
    CharField,
    FloatField,
    FloatVectorField,
    JsonField,
)


# 定义模型
class Product(Model):
    """产品模型示例"""

    name = CharField(max_length=255)
    description = CharField(max_length=1000)
    price = FloatField()
    # 为了简单示例，使用2维向量
    vector = FloatVectorField(dim=2)
    metadata = JsonField(nullable=True)

    class Meta:
        collection_name = "example_products"


async def main():
    print("=== Milvus ORM 示例 ===")

    # 1. 连接到Milvus
    print("\n1. 连接到Milvus...")
    try:
        await connect(uri="http://localhost:19530", token="root:QQNN_Milvus")
        print("✓ 成功连接到Milvus")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("请确保Milvus服务正在运行")
        return

    try:
        # 2. 创建集合
        print("\n2. 创建集合...")
        if await Product.create_collection():
            print("✓ Product集合创建成功")
        else:
            print("⚠ Product集合已存在")

        # 3. 插入数据
        print("\n3. 插入数据...")

        # 单个插入
        product1 = Product(
            name="智能手表",
            description="支持心率监测和GPS定位的智能手表",
            price=999.99,
            vector=[0.1, 0.2],
            metadata={"brand": "TechWatch", "color": "黑色", "stock": 100},
        )
        await product1.save()
        print("✓ 单个产品插入成功")
        print(product1.id)

        # 批量插入
        products = [
            Product(
                name="无线耳机",
                description="主动降噪无线蓝牙耳机",
                price=599.99,
                vector=[0.3, 0.4],
                metadata={"brand": "SoundBeats", "color": "白色", "stock": 200},
            ),
            Product(
                name="平板电脑",
                description="10英寸高清平板电脑",
                price=2999.00,
                vector=[0.5, 0.6],
                metadata={"brand": "TabTech", "color": "银色", "stock": 50},
            ),
        ]
        count = await Product.bulk_create(products)
        print(f"✓ 批量插入成功，创建了 {count} 个产品")

        # 4. 查询数据
        print("\n4. 查询数据...")

        # 获取单个对象
        try:
            found_product = await Product.objects.get(id=product1.id)
            print(f"✓ 获取单个产品: {found_product.name} - ¥{found_product.price}")
        except DoesNotExist:
            print("✗ 产品不存在")

        # 过滤查询
        expensive_products = await Product.objects.filter("price > 1000").all()
        print(f"✓ 价格大于1000的产品数量: {len(expensive_products)}")
        for p in expensive_products:
            print(f"  - {p.name}: ¥{p.price}")

        # 计数
        total_products = await Product.objects.count()
        print(f"✓ 产品总数: {total_products}")

        # 向量搜索示例
        query_vector = [0.2, 0.3]  # 接近智能手表的向量
        print(f"\n5. 向量搜索（查询向量: {query_vector}）...")
        similar_products = (
            await Product.objects.search(
                vector=query_vector,
                field_name="vector",
                metric_type="L2",  # 使用L2距离
                limit=2,
            )
            .limit(2)
            .all()
        )

        print(f"✓ 找到 {len(similar_products)} 个相似产品")
        for i, p in enumerate(similar_products):
            print(f"  {i + 1}. {p.name} - {p.description}")

        # 6. 更新数据
        print("\n6. 更新数据...")
        product_to_update = await Product.objects.get(id=product1.id)
        res = await product_to_update.update(
            price=899.99,
            name="智能手表Pro",
            metadata={"brand": "TechWatch", "color": "黑色", "stock": 95, "new": True},
        )
        print("✓ 产品更新成功", res)

        # 验证更新
        updated_product = await Product.objects.get(id=product1.id)
        print(f"  更新后: {updated_product.name} - ¥{updated_product.price}")

        # 7. 删除数据
        print("\n7. 删除数据...")
        # 删除单个产品
        await updated_product.delete()
        print("✓ 单个产品删除成功")

        # 批量删除
        delete_count = await Product.objects.filter("price < 1000").delete()
        print(f"✓ 批量删除成功，删除了 {delete_count} 个产品")

        # 验证删除后的数量
        remaining_products = await Product.objects.count()
        print(f"✓ 删除后剩余产品数量: {remaining_products}")

    finally:
        # 清理：删除集合（仅用于示例）
        print("\n8. 清理示例数据...")
        await Product.drop_collection()

        # 断开连接
        await disconnect()
        print("✓ 已断开与Milvus的连接")


if __name__ == "__main__":
    asyncio.run(main())
