from openai import OpenAI

# 1. 创建客户端，配置保持不变，仍然指向本地Ollama服务
client = OpenAI(
    base_url='http://10.20.62.189:11434/v1',
    api_key='ollama',
)

# 2. 发起聊天请求，只需将 model 参数修改为您新下载的模型名称
try:
    print("正在向本地模型 gpt-oss:20b 发送请求...")
    response = client.chat.completions.create(
      model="gpt-oss:20b",  # <--- 这里是您唯一需要修改的地方！
      messages=[
        {"role": "user", "content": "你好，请介绍一下你自己以及你的能力。"}
      ]
    )

    # 3. 打印模型的回答
    print("\n模型的回答是:\n")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\n发生错误: {e}")
    print("\n请确认以下几点：")
    print("1. Ollama 应用是否正在您的 Mac 上运行？")
    print("2. 您是否已经通过 `ollama run gpt-oss:20b` 或 `ollama pull gpt-oss:20b` 成功下载了该模型？")