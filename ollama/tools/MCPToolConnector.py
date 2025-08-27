# Located in src/pytorch/ollama/tools/MCPToolConnector.py
from langchain.agents import tool
import requests # 需要引入requests库
import json
from ollama.Constants import MCPConstant


@tool
def mcp_tool_connector(tool_call_request: str) -> dict:
    """
    一个特殊的工具，用于调用在MCP服务器上托管的任何工具。
    输入应该是一个JSON字符串，包含 'tool_id' 和 'parameters'。
    例如: '{{"tool_id": "get_sn_code_last_process_log", "parameters": {{"sn_code": "SN001"}}}}'
    """
    print(f"--- 📞 Agent is calling MCP Server with request: {tool_call_request} ---")
    try:
        # 解析模型生成的JSON字符串
        request_data = json.loads(tool_call_request)
        tool_id = request_data.get("tool_id")
        parameters = request_data.get("parameters")

        if not tool_id or not isinstance(parameters, dict):
            raise ValueError("输入必须是包含 'tool_id' 和 'parameters' 的JSON")

        # 向MCP服务器发送POST请求以执行工具
        response = requests.post(
            f"{MCPConstant.MCP_SERVER_URL.value}/mcp/tools_calls",
            json={"tool_id": tool_id, "parameters": parameters}
        )
        response.raise_for_status()# 如果请求失败则抛出异常
        return response.json()
    except Exception as e:
        return {"error": f"调用MCP工具失败: {str(e)}"}