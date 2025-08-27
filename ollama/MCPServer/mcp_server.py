import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

mock_process_pass_station_log = {
    "SN001": {"processCode": "PKG-WGJC", "testResult": "OK", "mftLine": "LINE-01", "passTime": "2025-08-18T10:00:00Z"},
    "SN002": {"processCode": "PKG-ZX", "testResult": "OK", "mftLine": "LINE-01", "passTime": "2025-08-18T11:30:00Z"},
    "SN003": {"processCode": "PKG-WGJC", "testResult": "NG", "mftLine": "LINE-02", "passTime": "2025-08-18T12:00:00Z"},
}


def get_sn_code_last_process_log(sn_code: str) -> dict:
    """
    根据提供的SN码，查询并返回其在生产过程中的最后一条过站记录。
    这个工具对于追踪一个特定单板的当前状态非常有用。
    """
    print(f"--- 🛠️ Tool Executing: get_sn_code_last_process_log with SN: {sn_code} ---")
    log_entry = mock_process_pass_station_log.get(sn_code)
    print(f"get_sn_code_last_process_log工具返回: {log_entry}")
    if log_entry:
        return log_entry
    else:
        return {"error": f"SN码 '{sn_code}' 没有找到任何过站记录。"}


AVAILABLE_TOOLS = {
    "get_sn_code_last_process_log": get_sn_code_last_process_log
}

app = FastAPI(
    title="My Custom Tools MCP Server",
    description="一个托管自定义工具以供AI Agent调用的MCP兼容服务",
)

# --- 3. 定义 MCP 协议需要的路由 ---

# MCP规范第一部分：工具清单 (让Agent知道这里有什么工具)
@app.get("/mcp/tools", tags=["MCP"])
def list_tools():
    """返回所有可用工具的清单，符合MCP规范。"""
    return {
        "tools": [
            {
                "id": "get_sn_code_last_process_log",
                "display_name": "查询SN码最后工序",
                "description": "根据提供的SN码，查询并返回其在生产过程中的最后一条过站记录。",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "sn_code": {
                            "type": "string",
                            "description": "需要查询的生产序列号，例如 'SN001'"
                        }
                    },
                    "required": ["sn_code"]
                }
            }
            # 如果您有更多工具，可以继续在这里添加
        ]
    }

# 定义工具调用的请求体
class ToolCallRequest(BaseModel):
    tool_id: str
    parameters: dict


# MCP规范第二部分：执行工具 (让Agent来调用工具)
@app.post("/mcp/tools_calls", tags=["MCP"])
async def call_tool(request: ToolCallRequest):
    """根据Agent的请求，执行指定的工具。"""
    tool_function = AVAILABLE_TOOLS.get(request.tool_id)
    if not tool_function:
        raise HTTPException(tatus_code=404, detail=f"Tool with id '{request.tool_id}' not found.")

    try:
        # 使用请求中的参数调用工具函数
        result = tool_function(**request.parameters)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing tool: {str(e)}")


# --- 4. 启动服务 ---
if __name__ == "__main__":
    print("启动MCP工具服务，访问 http://127.0.0.1:8000/docs 查看API文档")
    uvicorn.run(app, host="127.0.0.1", port=8000)

