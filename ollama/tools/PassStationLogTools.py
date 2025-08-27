from langchain.agents import tool

mock_process_pass_station_log = {
    "SN001": {"processCode": "PKG-WGJC", "testResult": "OK", "mftLine": "LINE-01", "passTime": "2025-08-18T10:00:00Z"},
    "SN002": {"processCode": "PKG-ZX", "testResult": "OK", "mftLine": "LINE-01", "passTime": "2025-08-18T11:30:00Z"},
    "SN003": {"processCode": "PKG-WGJC", "testResult": "NG", "mftLine": "LINE-02", "passTime": "2025-08-18T12:00:00Z"},
}


@tool
def get_sn_code_last_process_log(sn_code: str) -> dict:
    """
    根据提供的SN码，查询并返回其在生产过程中的最后一条过站记录。
    这个工具对于追踪一个特定单板的当前状态非常有用。
    """
    print(f"--- 🛠️ Tool Executing: get_sn_code_last_process_log with SN: {sn_code} ---")
    log_entry = mock_process_pass_station_log.get(sn_code)
    if log_entry:
        return log_entry
    else:
        return {"error": f"SN码 '{sn_code}' 没有找到任何过站记录。"}

