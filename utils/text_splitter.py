import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

# (关键) 建立文件后缀到 Language 枚举的映射
# 这让我们可以为 .py 文件调用 Python 分割器，为 .md 调用 Markdown 分割器
SUFFIX_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".md": Language.MARKDOWN,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    # .txt 和其他未知的，我们使用通用分割器
}

def split_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100
) -> List[Document]:
    """
    智能地分割文档列表。
    它会检查每个文档的元数据 (source)，
    并根据文件后缀选择特定于语言的分割器。

    :param documents: 从 loader 加载的 Document 列表
    :param chunk_size: 每个块的最大 Token 数（估算值）
    :param chunk_overlap: 块之间的重叠 Token 数
    :return: 分割后的 Document 块 (chunks) 列表
    """
    print(f"Splitting {len(documents)} documents...")
    
    split_chunks = []
    
    # 为我们不认识的语言创建一个“通用”分割器
    generic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    for doc in documents:
        file_path = doc.metadata.get("source")
        if not file_path:
            # 如果没有源文件路径，使用通用分割器
            split_chunks.extend(generic_splitter.split_documents([doc]))
            continue
        
        # 1. 从文件路径获取后缀
        _, file_ext = os.path.splitext(file_path)
        
        # 2. 查找对应的 Language
        lang = SUFFIX_TO_LANGUAGE.get(file_ext)
        
        if lang:
            # 3. (智能) 如果找到了对应的语言，使用 from_language 创建分割器
            try:
                lang_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                split_chunks.extend(lang_splitter.split_documents([doc]))
            except Exception as e:
                print(f"Could not use language splitter for {file_path}: {e}. Using generic splitter.")
                # 如果 from_language 失败（比如 lang lib 没装），退回到通用版
                split_chunks.extend(generic_splitter.split_documents([doc]))
        else:
            # 4. (通用) 如果是不支持的后缀 (如 .txt)，使用通用分割器
            split_chunks.extend(generic_splitter.split_documents([doc]))
            
    print(f"Splitting complete. Total chunks: {len(split_chunks)}")
    return split_chunks