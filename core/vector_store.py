from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

# 导入我们自己的 Embedder
from core.embedder import get_embedder

# (可选) 定义我们的 ChromaDB 存储路径
DEFAULT_CHROMA_PATH = "./chroma_db"

def create_vector_store(
    documents: List[Document], 
    persist_directory: str = DEFAULT_CHROMA_PATH
) -> VectorStore:
    """
    从文档块 (chunks) 创建并持久化一个向量存储。
    这将调用 ZhipuAI API 进行嵌入，并将结果存入 ChromaDB。

    :param documents: 已经分割好的 Document 块
    :param persist_directory: 持久化存储的本地路径
    :return: 一个 Chroma VectorStore 实例
    """
    print(f"Initializing embedder (ZhipuAI)...")
    embedder = get_embedder()
    
    print(f"Creating vector store at '{persist_directory}'...")
    # LangChain 的 from_documents 会自动处理：
    # 1. 遍历所有 documents
    # 2. 调用 embedder.embed_documents() (批量嵌入)
    # 3. 将 (chunk + vector) 存入 ChromaDB
    # 4. 持久化到磁盘
    vector_store = Chroma.from_documents(
        documents=documents, 
        embedding=embedder,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created successfully with {vector_store._collection.count()} entries.")
    return vector_store

def load_vector_store(
    persist_directory: str = DEFAULT_CHROMA_PATH
) -> VectorStore:
    """
    从磁盘加载一个已存在的向量存储。

    :param persist_directory: 存储的本地路径
    :return: 一个 Chroma VectorStore 实例
    """
    print(f"Loading vector store from '{persist_directory}'...")
    embedder = get_embedder() # 加载时也需要 embedder
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedder
    )
    
    print(f"Vector store loaded successfully with {vector_store._collection.count()} entries.")
    return vector_store

def get_retriever(
    vector_store: VectorStore, 
    top_k: int = 5
) -> BaseRetriever:
    """
    从一个 VectorStore 实例获取一个检索器 (Retriever)。

    :param vector_store: 已加载的 VectorStore
    :param top_k: 每次检索返回的相关文档数量
    :return: 一个 BaseRetriever 实例
    """
    print(f"Creating retriever with top_k={top_k}...")
    # as_retriever 是将数据库转换为“检索器”的标准方法
    return vector_store.as_retriever(search_kwargs={"k": top_k})