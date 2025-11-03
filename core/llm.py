# 导入的包变了：不再是 langchain_moonshot，而是 langchain_openai
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from core.config import get_kimi_api_key

def get_llm() -> BaseChatModel:
    """
    获取一个配置好的 Kimi (Moonshot) LLM 实例。
    
    我们使用 ChatOpenAI 类，并指定 Kimi 的 'base_url' 和 'model'
    来实现 OpenAI 兼容的 API 调用。
    
    :return: 一个实现了 BaseChatModel 接口的 Kimi LLM 实例。
    """
    
    llm = ChatOpenAI(
        # 1. (关键) 指定 Kimi 的服务器地址
        base_url="https://api.moonshot.cn/v1",
        
        # 2. (关键) 传入 Kimi 的 API Key
        api_key=get_kimi_api_key(),
        
        # 3. 指定 Kimi 的模型名称
        model_name="kimi-k2-turbo-preview", 
        
        # 4. 设置 temperature=0.0 以获得确定性回答
        temperature=0.0
    )
    return llm

# 自我测试 (这段代码不需要变)
if __name__ == "__main__":
    print("--- 正在测试 Kimi LLM 初始化 (使用 OpenAI 兼容模式) ---")
    try:
        llm = get_llm()
        print(f"LLM 实例创建成功: {type(llm)}")
        
        # 发起一个简单的调用来测试 API Key 和连接
        print("正在测试 API 调用...")
        
        # LangChain 会自动将 "你好，Kimi！" 转换为 OpenAI 格式的 JSON
        # 并发送到 https://api.moonshot.ai/v1
        response = llm.invoke("你好，Kimi！") 
        
        print(f"Kimi 回答: {response.content}")
        print("Kimi LLM 模块测试通过！")
    except Exception as e:
        print(f"Kimi LLM 模块测试失败: {e}")