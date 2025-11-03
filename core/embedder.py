from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.embeddings import Embeddings
from core.config import get_zhipu_api_key

def get_embedder() -> Embeddings:
    """
    获取一个配置好的 ZhipuAI (智谱) Embedder 实例。
    
    使用 Embeddings 作为返回类型提示，同样是“面向接口编程”的风格。
    我们使用默认的 embedding-2 模型。

    :return: 一个实现了 Embeddings 接口的 Zhipu Embedder 实例。
    """
    
    embedder = ZhipuAIEmbeddings(
        api_key=get_zhipu_api_key()
        # 我们使用默认的 embedding-2 模型，因此无需额外指定
    )
    return embedder

# 自我测试
if __name__ == "__main__":
    print("--- ZhipuAI Embedder 初始化 ---")
    try:
        embedder = get_embedder()
        print(f"Embedder 实例创建成功: {type(embedder)}")
        
        # 测试将一段文本转换为向量
        print("正在测试文本嵌入...")
        test_text = "这是我们的LLM Agent项目"
        vector = embedder.embed_query(test_text)
        
        print(f"成功生成向量！维度 (Dimensions): {len(vector)}")
        print(f"向量前5位预览: {vector[:5]}")
        print("ZhipuAI Embedder 模块测试通过！")
    except Exception as e:
        print(f"ZhipuAI Embedder 模块测试失败: {e}")