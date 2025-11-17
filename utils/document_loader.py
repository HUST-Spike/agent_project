import os
from typing import List
from pathlib import Path
import glob
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.documents import Document

# 定义我们关心的文件类型
# 你可以根据需要添加更多，比如 .java, .cpp, .html 等
SUPPORTED_EXTENSIONS = [
    ".py",
    ".js",
    ".ts",
    ".md",
    ".txt",
    ".go",
    ".java",
    ".cpp",
    ".h",
    ".html",
    ".css",
    ".c",
]

def load_code_from_path(repo_path: str) -> List[Document]:
    """
    从本地仓库路径加载所有支持的代码和文档文件。

    :param repo_path: 本地仓库的根路径
    :return: LangChain Document 列表
    """
    if not os.path.isdir(repo_path):
        raise ValueError(f"Path '{repo_path}' is not a valid directory.")
        
    print(f"Loading documents from {repo_path}...")
    
    # 我们优先使用 GenericLoader 来处理复杂的代码库并使用 LanguageParser
    # 注意：在某些环境下（尤其是 Windows），mimetype 检测可能失败（缺少 libmagic），
    # 导致 GenericLoader 抛出 "does not have a mimetype" 的错误。为提高健壮性，
    # 我们在 loader.load() 失败时回退到按后缀手动读取文件内容。
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=SUPPORTED_EXTENSIONS,
        exclude=[
            "**/__pycache__/*",
            "**/.git/*",
            "**/.venv/*",
            "**/node_modules/*",
            "**/*.lock",
            "**/*.log",
        ],
        show_progress=True,
    )

    try:
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents
    except ValueError as e:
        # 处理 GenericLoader 在某些平台上无法识别 mimetype 的情况，回退到简单读取文本文件
        print(f"GenericLoader failed with error: {e}. Falling back to simple file read.")
        docs = []
        # 使用 pathlib / glob 遍历所有匹配的后缀文件
        repo_path_obj = Path(repo_path)
        patterns = [f"**/*{ext}" for ext in SUPPORTED_EXTENSIONS]
        seen = set()
        for pattern in patterns:
            for file_path in repo_path_obj.glob(pattern):
                try:
                    # 跳过目录
                    if file_path.is_dir():
                        continue
                    # 避免重复
                    pstr = str(file_path.resolve())
                    if pstr in seen:
                        continue
                    seen.add(pstr)
                    # 以文本方式读取（忽略二进制或编码错误）
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": pstr}))
                except Exception:
                    # 忽略无法读取的文件
                    continue
        print(f"Fallback loaded {len(docs)} documents.")
        return docs