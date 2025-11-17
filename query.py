import os
from dotenv import load_dotenv
load_dotenv() 
from core.rag_chain import create_rag_chain

# --- (调试步骤) 在这里设置你的问题 ---
TEST_QUESTION = "数据库是如何实现的？它在哪个文件里定义的？"


def main():
    """
    执行 RAG 查询的主函数。
    """
    print("--- [Query Script] Starting... ---")
    
    # 确保 ZHIPUAI_API_KEY 和 MOONSHOT_API_KEY 已设置
    if not os.environ.get("ZHIPUAI_API_KEY") or not os.environ.get("MOONSHOT_API_KEY"):
        print("Error: ZHIPUAI_API_KEY or MOONSHOT_API_KEY not found in .env file.")
        print("Please ensure your .env file is correctly set up in the root directory.")
        return

    try:
        # 1. 创建 RAG 链
        # (这会加载 LLM, ChromaDB, Retriever 等)
        rag_chain = create_rag_chain()
        
        # 2. (核心) 执行查询
        print(f"\n--- [Query Script] Executing RAG chain for question ---")
        print(f"Question: {TEST_QUESTION}")
        print("\n--- [Kimi's Answer] ---")
        
        # .stream() 是一个更高级的用法，它会流式打印答案
        # .invoke() 会等待 Kimi 完全回答后再打印
        
        # 我们用 .stream() 来获得类似 ChatGPT 的打字机效果
        full_response = ""
        for chunk in rag_chain.stream(TEST_QUESTION):
            print(chunk, end="", flush=True) # 实时打印 Kimi 返回的“块”
            full_response += chunk
        
        print("\n\n--- [Query Script] Execution finished. ---")

    except Exception as e:
        print(f"\n--- [FAILED] An error occurred during query: {e} ---")

if __name__ == "__main__":
    main()