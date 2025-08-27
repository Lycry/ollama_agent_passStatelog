import os
import httpx
from langchain_community.chat_models import ChatOllama
from ollama.tools.PassStationLogTools import get_sn_code_last_process_log
from ollama.tools.MCPToolConnector import mcp_tool_connector
from langchain.retrievers.multi_query import MultiQueryRetriever
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests
from ollama.Constants import MCPConstant

from langchain_core.exceptions import OutputParserException
import json
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
import operator
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction, AgentFinish

# --- 您的本地Ollama IP地址 (如果 'localhost' 不行) ---
OLLAMA_IP = "127.0.0.1"  # 通常 '127.0.0.1' 或 'localhost' 都可以
# ----------------------------------------------------

# --- 1. 初始化模型和嵌入 ---
# 确保Ollama服务正在运行
try:
    client = OpenAI(
        base_url=f'http://{OLLAMA_IP}:11434/v1',
        api_key='ollama',
        http_client=httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))
    )
    # 检查连接
    client.models.list()
    print("成功连接到Ollama服务。")
except Exception as e:
    print(f"无法连接到Ollama服务，请确认它正在运行并且IP地址正确: {e}")
    exit()

# 使用您本地的gpt-oss:20b模型
llm = ChatOllama(model="gpt-oss:20b", base_url=f'http://{OLLAMA_IP}:11434')

# 使用一个专门的嵌入模型将文本转换为向量
# 在终端运行: ollama pull nomic-embed-text
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=f'http://{OLLAMA_IP}:11434')

print("模型和嵌入模块初始化成功...")

# --- 2. 加载和处理代码文件 ---
code_directory = "/Users/hm/mycode/qlled-edge-sys/qlled-edge-modules/qlled-edge-package"  # 存放您所有.java文件的文件夹

if not os.path.isdir(code_directory):
    print(f"错误: 找不到代码目录 '{code_directory}'。")
    exit()

# 使用DirectoryLoader加载目录下所有.java文件
loader = DirectoryLoader(code_directory, glob="**/*.java", loader_cls=TextLoader)
documents = loader.load()

if not documents:
    print(f"警告: 在 '{code_directory}' 目录中没有找到任何.java文件。")
    exit()

# 针对Java代码进行语义分割
java_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=1500, chunk_overlap=200
)
splits = java_splitter.split_documents(documents)

print(f"代码库已加载并分割成 {len(splits)} 个代码块。")

# --- 3. 创建向量数据库 ---
# 将分割后的代码块存入ChromaDB
dbPath = "/Users/hm/PycharmProjects/pytorch/data/chroma_db"
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=dbPath)

# 创建一个检索器
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10}),  # 检索结果top10
    llm=llm
)

print("代码向量数据库创建成功！")

# # --- 创建 RAG 链 ---
# template = """
# 你是一个资深的Java软件架构师。请根据下面提供的代码上下文，专业地回答用户的问题。
# 请直接了当地回答，并在必要时引用关键的类名或方法名。如果上下文中没有足够信息，请说明无法回答。
#
# 代码上下文:
# {context}
#
# 问题:
# {question}
#
# 专业回答:
# """
# prompt = ChatPromptTemplate.from_template(template)
#
# output_parser = StrOutputParser()
#
# # 构建处理链
# rag_chain = (
#         {"context": retriever_from_llm, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | output_parser
# )

print("代码问答机器人已准备就绪！")

# --- 更新Agent的工具列表 ---
# 首先，从MCP服务器动态获取工具列表及其描述
try:
    print("正在连接mcpServer。。。")
    print(f"{MCPConstant.MCP_SERVER_URL.value}/mcp/tools")
    mcp_tools_manifest = requests.get(f"{MCPConstant.MCP_SERVER_URL.value}/mcp/tools").json()["tools"]
    # 将MCP工具的描述动态地添加到我们的Prompt中，让Agent知道它们的存在
    mcp_tools_description = "\n".join([
        f"- Tool ID: `{tool['id']}`, Description: {tool['description']}, Parameters: {tool['parameters_schema']}"
        for tool in mcp_tools_manifest
    ])
except Exception as e:
    print(f"警告：无法从MCP服务器获取工具列表: {e}")
    mcp_tools_description = "没有可用的MCP工具。"

code_retriever_tool = create_retriever_tool(
    retriever_from_llm,
    "code_search",
    "Searches and returns code snippets from the qlled-edge-package Java codebase."
)


tools = [code_retriever_tool, mcp_tool_connector]

# --- 更新Agent的Prompt，告诉它如何使用MCP工具 ---
# 我们需要一个更明确的Prompt来引导模型正确地调用MCP工具
# 这是对您之前使用的ReAct Prompt的增强
# 通过字符串拼接和替换来构建最终的Prompt，以避免f-string的解析问题
# 这能确保mcp_tools_description中的JSON schema被正确转义
react_prompt_template_str = (
    "尽你所能回答以下问题。你可以使用以下工具：\n\n"
    "{tools}\n\n"
    "另外，你还可以通过 'mcp_tool_connector' 工具访问一个远程MCP服务。 以下是远程MCP服务上可用的工具列表:\n"
    + mcp_tools_description.replace("{", "{{").replace("}", "}}") + "\n"
    "当你需要使用这些远程工具时，你必须调用 'mcp_tool_connector' 工具，\n"
    "并且 Action Input 必须是严格的JSON格式，包含 \"tool_id\" 和 \"parameters\"。\n\n"
    "请严格使用以下格式：\n\n"
    "Question: 你必须回答的输入问题\n"
    "Thought: 你应该时刻思考该做什么。\n"
    "Action: 要采取的行动，应该是[{tool_names}]中的一个。\n"
    "Action Input: 采取行动的输入。\n"
    "Observation: 行动的结果\n"
    "Thought: 我现在知道最终答案了\n"
    "Final Answer: 对原始输入问题的最终回答\n\n"
    "--- 这是一个完整的例子 ---\n"
    "Question: SN002现在在哪里？\n"
    "Thought: 我需要查询SN002的过站记录，我应该使用`get_sn_code_last_process_log`工具，并通过`mcp_tool_connector`来调用。\n"
    "Action: mcp_tool_connector\n"
    "Action Input: {{\"tool_id\": \"get_sn_code_last_process_log\", \"parameters\": {{\"sn_code\": \"SN002\"}}}}\n"
    "Observation: {{'result': {{'processCode': 'PKG-ZX', 'testResult': 'OK', 'mftLine': 'LINE-01', 'passTime': "
                                                                    "'2025-08-18T11:30:00Z'}}}}\n"
    "Thought: 我已经从工具得到了SN002的最后过站记录，现在可以总结并回答用户的问题了。\n"
    "Final Answer: SN002的最后位置是在LINE-01产线的PKG-ZX工站，通过时间是2025-08-18T11:30:00Z，测试结果为OK。\n"
    "--- 例子结束 ---\n\n"
    "现在，开始！\n\n"
    "Question: {input}\n"
    "Thought:{agent_scratchpad}"
)


# agent_prompt = hub.pull("hwchase17/react")
agent_prompt = PromptTemplate.from_template(react_prompt_template_str)

# agent = create_react_agent(llm, tools, agent_prompt)
#
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
#
# print("✅ 代码助理Agent已成功创建并配备工具！")
#
# # --- 6. 与您的Agent互动 ---
# if __name__ == "__main__":
#     print("\n--- qlled-edge-package 智能代码助理 ---")
#     print("您可以提出问题或下达指令 (输入 'exit' 退出):")
#
#     while True:
#         user_input = input("\n> ")
#         if user_input.lower() == 'exit':
#             break
#
#         # 将用户的输入交给Agent执行器
#         result = agent_executor.invoke({"input": user_input})
#
#         print("\n助理:\n", result["output"])
#
#     print("\n感谢使用，再见！")

# LangGraph 改造：创建一个 ToolExecutor，这是 LangGraph 调用工具的标准方式

# --- 5. 创建 LangGraph Agent ---

# 定义 Agent 状态
# AgentState 是一个 TypedDict，它定义了在图的节点之间传递的状态。


class AgentState(TypedDict):
    input: str
    chat_history: Sequence[BaseMessage]
    # an agent_outcome can be an AgentAction or AgentFinish
    agent_outcome: Union[AgentAction, AgentFinish]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# 定义 Agent 和图的节点
# 使用与之前相同的 ReAct prompt
# agent_prompt = hub.pull("hwchase17/react")
# 创建 ReAct Agent 的核心逻辑
agent_runnable = create_react_agent(llm, tools, agent_prompt)


# 定义图的 "大脑" 节点：决定下一步做什么
def run_agent(state: AgentState):
    """
    运行Agent的核心逻辑，并增加了错误处理机制。
    此版本能捕捉到模型返回空输出时的失败，并进行优雅地处理，而不是让程序崩溃。
    """
    try:
        agent_outcome = agent_runnable.invoke(state)
        return {"agent_outcome": agent_outcome}
    except OutputParserException as e:
        # 从异常信息中提取模型最原始的输出文本
        raw_output = str(e).split("Could not parse LLM output: `")[-1].strip().rstrip('`')

        # 新增：检查原始输出是否为空。这是导致程序崩溃的主要原因。
        if not raw_output:
            # 如果模型什么都没返回，说明它卡住了。
            # 我们将手动创建一个AgentFinish对象，来平稳地终止流程。
            error_message = "抱歉，我似乎在处理您的请求时遇到了问题。您能换一种方式提问吗？"
            print(f"--- ⚠️ 恢复模式：模型返回了空字符串。强制生成最终答案。 ---")
            return {
                "agent_outcome": AgentFinish(
                    return_values={"output": error_message},
                    log="由于LLM输出为空，强制结束。"
                )
            }

        # 这是原有的恢复机制：当工具被调用后，模型未能格式化最终答案时触发。
        if state.get("intermediate_steps"):
            print("--- ⚠️ 恢复模式：模型在工具调用后未能生成最终答案。正在总结结果。 ---")
            last_tool_output = state["intermediate_steps"][-1][1]
            original_question = state["input"]

            final_answer_prompt = PromptTemplate.from_template(
                """根据用户的问题和从工具检索到的数据，提供一个直接的、最终的中文回答。

                用户问题: {question}
                检索到的数据: {context}

                最终答案:"""
            )
            final_answer_chain = final_answer_prompt | llm | StrOutputParser()
            final_answer = final_answer_chain.invoke({
                "question": original_question,
                "context": str(last_tool_output)
            })
            return {
                "agent_outcome": AgentFinish(
                    return_values={"output": final_answer},
                    log="恢复成功。已根据上下文生成最终答案。"
                )
            }

        # 如果所有恢复条件都不满足，则重新抛出异常。
        raise e


# 定义图的条件边
# 这个函数决定在 'agent' 节点之后应该走哪条路
def should_continue(state: AgentState):
    """
    根据 Agent 的输出来决定流程。
    - 如果是 AgentFinish，则结束。
    - 如果是 AgentAction，则调用工具。
    """
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"


def execute_tools(state: AgentState) -> dict:
    """
    一个自定义的工具执行节点。
    它会从 state 中获取 AgentAction，执行相应的工具，
    然后将结果格式化为 ReAct Agent 下一步所期望的格式。
    """
    agent_action = state.get("agent_outcome")

    # 确保我们有一个有效的 AgentAction
    if not isinstance(agent_action, AgentAction):
        raise ValueError("在状态中没有找到有效的 AgentAction。")

    # 获取工具名称和输入
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    # 找到要执行的工具函数
    tool_to_execute = None
    for tool in tools:
        if tool.name == tool_name:
            tool_to_execute = tool
            break

    # 执行工具并获取结果
    if not tool_to_execute:
        observation = f"错误: 找不到名为 '{tool_name}' 的工具。"
    else:
        try:
            # 调用工具并获取其输出
            observation = tool_to_execute.invoke(tool_input)
        except Exception as e:
            observation = f"执行工具 '{tool_name}' 时出错: {e}"

    # ReAct Agent 期望将 (AgentAction, observation) 元组添加到 intermediate_steps
    return {
        "intermediate_steps": [(agent_action, str(observation))]
    }


# 组装图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)

# 设置入口点
workflow.set_entry_point("agent")

# 设置条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)


# 添加从 action 回到 agent 的边
workflow.add_edge("action", "agent")

# 编译图，生成可执行的 app
app = workflow.compile()
print("✅ LangGraph Agent 已成功编译！")

# --- 与您的 LangGraph Agent 互动 ---
if __name__ == "__main__":
    print("\n--- qlled-edge-package 智能代码助理 (LangGraph版) ---")
    print("您可以提出问题或下达指令 (输入 'exit' 退出):")

    while True:
        user_input = input("\n> ")
        if user_input.lower() == 'exit':
            break

        # 使用 LangGraph app 来处理输入
        # 我们传入一个符合 AgentState 结构的字典
        inputs = {"input": user_input, "chat_history": [], "intermediate_steps": []}

        # LangGraph 会自动循环直到 'end' 状态
        result = app.invoke(inputs)

        # 最终结果在 agent_outcome 字段中
        final_output = result["agent_outcome"].return_values["output"]

        print("\n助理:\n", final_output)

    print("\n感谢使用，再见！")
