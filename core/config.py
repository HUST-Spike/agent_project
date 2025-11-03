import os
from dotenv import load_dotenv

# 在模块被导入时，自动加载 .env 文件中的环境变量
# 这确保了只要有任何模块导入 config，API Keys 就会被加载到环境中
load_dotenv()

def get_kimi_api_key() -> str:
    """
    获取 Kimi (Moonshot) API Key。
    
    :return: API Key 字符串
    :raises ValueError: 如果环境变量中未找到 API Key
    """
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError(
            "MOONSHOT_API_KEY 未设置在环境变量中。"
            "请检查你的 .env 文件。"
        )
    return api_key

def get_zhipu_api_key() -> str:
    """
    获取 ZhipuAI (智谱) API Key。
    
    :return: API Key 字符串
    :raises ValueError: 如果环境变量中未找到 API Key
    """
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError(
            "ZHIPUAI_API_KEY 未设置在环境变量中。"
            "请检查你的 .env 文件。"
        )
    return api_key


if __name__ == "__main__":
    print("--- 正在测试配置加载 ---")
    try:
        kimi_key = get_kimi_api_key()
        print(f"Kimi Key (后4位): ...{kimi_key[-4:]}")
        
        zhipu_key = get_zhipu_api_key()
        print(f"Zhipu Key (后4位): ...{zhipu_key[-4:]}")
        
        print("配置加载成功！")
    except ValueError as e:
        print(f"配置加载失败: {e}")