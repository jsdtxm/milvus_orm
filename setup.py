from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="milvus_orm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Django-style ORM for Milvus vector database with async support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/milvus_orm",
    packages=find_packages(),
    package_data={
        "milvus_orm": ["py.typed"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "pymilvus>=2.3.0",
    ],
)