import os
import shutil
import stat

# 导入我们刚刚创建的所有模块
from utils.git_loader import clone_repo
from utils.document_loader import load_code_from_path
from utils.text_splitter import split_documents
from core.vector_store import create_vector_store

def remove_readonly(func, path, exc_info):
    """
    错误处理函数，用于处理 Windows 上删除只读文件的问题。
    当 shutil.rmtree 遇到权限错误时，会调用此函数。
    """
    # 移除只读属性并重试删除
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_rmtree(path):
    """
    安全地删除目录树，处理 Windows 上的只读文件和权限问题。
    """
    if os.path.exists(path):
        try:
            # Windows 上需要传入 onerror 来处理只读文件
            shutil.rmtree(path, onerror=remove_readonly)
        except Exception as e:
            print(f"Warning: Could not fully remove {path}: {e}")
            # 即使删除失败也继续执行，避免阻塞流程

# --- 配置你的仓库 ---
# (你可以换成任何你想分析的仓库)
# REPO_URL = "https://github.com/langchain-ai/langchain.git"
REPO_URL = "https://github.com/HUST-Spike/TodoList.git" # 自己仓库里的测试项目

# 定义本地路径
REPO_PATH = "./repos/"
CHROMA_PATH = "./chroma_db/"

def main():
    """
    执行完整的数据灌入 (Ingestion) 流程。
    """
    print("--- [Step 1/4] Starting Ingestion Pipeline ---")
    
    # (可选) 在开始前，清空旧数据
    safe_rmtree(REPO_PATH)
    safe_rmtree(CHROMA_PATH)
        
    try:
        # 步骤 1: 克隆仓库
        print("\n--- [Step 2/4] Cloning Repository ---")
        clone_repo(REPO_URL, REPO_PATH)
        
        # 步骤 2: 加载文件
        print("\n--- [Step 3/4] Loading Documents ---")
        documents = load_code_from_path(REPO_PATH)
        
        # 步骤 3: 智能分割
        print("\n--- [Step 4/4] Splitting Documents ---")
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)
        
        # 步骤 4: 嵌入并存储 (这是最花时间的一步)
        print("\n--- [Step 5/5] Creating Vector Store ---")
        print("(This may take several minutes depending on repo size and API speed...)")
        create_vector_store(chunks, CHROMA_PATH)
        
        print("\n--- [SUCCESS] Ingestion Pipeline Completed! ---")
        print(f"Vector data stored in: {CHROMA_PATH}")

    except Exception as e:
        print(f"\n--- [FAILED] An error occurred: {e} ---")
        # (可选) 清理失败的残留文件
        safe_rmtree(REPO_PATH)
        safe_rmtree(CHROMA_PATH)

if __name__ == "__main__":
    # 确保你的 .env 文件里有 ZHIPUAI_API_KEY
    main()