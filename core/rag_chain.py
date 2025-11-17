from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 导入我们已有的核心模块
from core.llm import get_llm
from core.vector_store import load_vector_store, get_retriever

# 定义 RAG 提示词模板
# 这是 RAG 的“灵魂”，它指导 LLM 如何利用上下文
RAG_PROMPT_TEMPLATE = """
你是一个资深的 Python 程序员和代码库助手。
请根据下面提供的 [上下文代码](Context)，用中文清晰地回答 [问题](Question)。
只使用 [上下文代码] 中的信息来回答问题。如果你在上下文中找不到答案，请明确地说：“根据提供的上下文，我无法回答这个问题。”

[上下文代码]:
{context}

[问题]:
{question}
"""

def _format_docs(docs: list[Document]) -> str:
    """
    一个辅助函数，用于将检索到的 Document 列表格式化为纯字符串。
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def create_rag_chain() -> Runnable:
    """
    创建并返回一个完整的 RAG (Retrieval-Augmented Generation) 链。

    这条链会执行以下操作：
    1. 并行地：
        a. 检索上下文 (问题 -> Retriever -> 格式化)
        b. 传递原始问题
    2. 将检索到的上下文和问题插入到提示词模板中
    3. 将填充好的提示词发送给 LLM (Kimi)
    4. 解析 LLM 的输出为字符串

    :return: 一个可运行的 (Runnable) RAG 链
    """
    print("--- [RAG Chain] Initializing components... ---")
    
    # 1. 加载 LLM (Kimi)
    llm = get_llm()
    print("--- [RAG Chain] LLM (Kimi) loaded. ---")

    # 2. 加载向量数据库 (ChromaDB)
    # (注意：这里我们硬编码了路径，这在模块中是合理的)
    try:
        vector_store = load_vector_store(persist_directory="./chroma_db/")
    except Exception as e:
        print(f"[RAG Chain] Fatal Error: Failed to load vector store from './chroma_db/'.")
        print(f"Did you run 'python ingest.py' successfully first?")
        raise e
        
    print("--- [RAG Chain] Vector Store loaded. ---")

    # 3. 从数据库创建检索器
    # (我们检索 5 个最相关的代码块)
    retriever = get_retriever(vector_store, top_k=5)
    print("--- [RAG Chain] Retriever created. ---")

    # 4. 创建提示词模板
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # 5. (核心) 构建 LCEL (LangChain Expression Language) 链
    
    # 5a. 创建一个并行处理步骤，它会：
    #    - (1) 调用检索器获取上下文
    #    - (2) 传递原始问题
    # RunnableParallel (或 RunnableMap) 允许我们并行执行
    setup_and_retrieval = RunnableParallel(
        {
            "context": retriever | _format_docs, # "context" 键的值是 (retriever | _format_docs) 链的结果
            "question": RunnablePassthrough() # "question" 键的值是原始输入 (问题字符串)
        }
    )

    # 5b. 将所有组件“粘合”在一起
    rag_chain = (
        setup_and_retrieval # 第一步：输入问题，输出 {"context": ..., "question": ...}
        | prompt            # 第二步：将字典填充到提示词中
        | llm               # 第三步：将提示词发送给 Kimi
        | StrOutputParser() # 第四步：将 Kimi 的 ChatMessage 回答解析为纯字符串
    )
    
    print("--- [RAG Chain] RAG chain created successfully. ---")
    return rag_chain