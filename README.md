PyTorch 与 Transformer 深度学习实践项目
这是一个综合性的项目，旨在探索与实践 PyTorch、Hugging Face Transformers 以及基于大型语言模型（如 Ollama）的应用。项目包含了从基础的 PyTorch 操作、自定义数据集处理，到复杂的 Transformer 模型实现，最终到一个整合了本地大型语言模型、向量数据库和外部工具的智能代码问答 Agent。

项目结构概览
.
└── src/
    └── pytorch/
        ├── ollama/                  # Ollama 本地化大模型应用相关
        │   ├── MCPServer/
        │   │   └── mcp_server.py    # 一个 FastAPI 服务器，用于提供外部工具
        │   ├── tools/
        │   │   ├── MCPToolConnector.py  # 连接 MCP 服务器的工具
        │   │   └── PassStationLogTools.py # 示例工具：查询生产日志
        │   ├── Constants.py         # 项目中使用的常量
        │   ├── gpt_oss.py           # 测试本地 Ollama 模型的脚本
        │   └── gpt_oss_code_qa_app.py # 核心应用：基于代码库的问答 Agent
        │
        ├── pariwise_cls_similarity_afqmc/ # AFQMC 数据集上的成对句子相似度任务
        │   ├── AFQMC.py             # AFQMC 的标准 Dataset 实现
        │   ├── AFQMCDataLoader.py   # 为 AFQMC 数据集创建 DataLoader
        │   ├── BertForPairwiseCLS.py# 用于句子对分类的 Bert 模型
        │   └── IterableAFQMC.py     # AFQMC 的可迭代 Dataset 实现
        │
        └── test/                      # PyTorch 和 Transformers 的基础测试与示例
            ├── CustomImageDataset.py  # 自定义图片数据集的示例
            ├── ModelTest.py           # Hugging Face AutoModel 和 AutoTokenizer 基础用法
            ├── Multi_Head_Attention.py# 从零实现多头注意力机制
            ├── MyIterableDataSet.py   # 自定义可迭代数据集的示例
            ├── PytorchTest.py         # PyTorch 基础张量操作和 GPU 测试
            ├── SampleDataSet.py       # 使用 FashionMNIST 进行模型训练和评估的完整示例
            ├── Single_Attention.py    # 从零实现缩放点积注意力
            └── TransfomerTest.py      # Hugging Face Pipeline API 的用法展示
各组件详细介绍
1. Ollama 智能代码问答 Agent (ollama/)
这是本项目的核心应用，展示了如何利用本地大语言模型（LLM）、LangChain 框架和向量数据库来构建一个能够理解并回答关于特定代码库问题的智能助理。它采用了 LangGraph 来构建一个稳定且具备错误恢复能力的 Agent。

gpt_oss_code_qa_app.py: 这是 Agent 的主应用程序。它完成了以下工作：

模型与嵌入初始化：加载本地 Ollama 中的大语言模型 (gpt-oss:20b) 和嵌入模型 (nomic-embed-text)。

文件加载与分割：读取指定目录下的所有 .java 代码文件，并使用针对 Java 语言的分割器将代码切分成有意义的区块。

向量数据库创建：将分割后的代码块转换为向量，并存储在 ChromaDB 中，以便进行快速的语义搜索。

工具定义：定义了两种工具：一个用于在向量数据库中检索代码的 code_search 工具，以及一个用于调用外部 API 的 mcp_tool_connector 工具。

Agent 与图 (Graph) 的创建：使用 LangGraph 框架来定义 Agent 的工作流程。这比传统的 AgentExecutor 更稳定，它将 Agent 的思考 -> 行动 -> 观察的循环定义为一个状态图，包含了错误处理和恢复机制。

交互式界面：提供一个命令行界面，让用户可以与 Agent 进行互动问答。

MCPServer/mcp_server.py: 一个使用 FastAPI 创建的轻量级网页服务器。它的作用是将本地的 Python 函数（例如查询生产日志）封装成符合 MCP (Machine-readable Capability Protocol) 规范的 API 接口，让 Agent 可以通过网络调用这些外部工具。

tools/: 这个目录存放了 Agent 可以使用的工具。

PassStationLogTools.py: 定义了一个具体的工具 get_sn_code_last_process_log，用于模拟查询一个产品的生产日志。这是 Agent 解决特定领域问题能力的来源。

MCPToolConnector.py: 一个通用的 LangChain 工具，它的唯一作用是作为桥梁，去调用远程 mcp_server.py 上托管的任何工具。Agent 通过它来执行例如查询日志等远程操作。

Constants.py: 定义了项目中会用到的常量，例如服务器 URL 和 HTTP 状态码，便于管理与修改。

gpt_oss.py: 一个简单的测试脚本，用于直接与本地 Ollama 模型进行对话，以验证模型是否正常运作。

2. 句子相似度分类任务 (pariwise_cls_similarity_afqmc/)
这个模块专注于一个经典的自然语言处理任务：判断两个句子语义是否相同。它使用了 AFQMC (Ant Financial Question Matching Corpus) 数据集，并基于 BERT 模型进行微调。

AFQMC.py & IterableAFQMC.py: 分别用两种方式实现了 PyTorch 的数据集类别。

AFQMC.py: 使用标准的 Dataset，一次性将所有数据读取到内存中。

IterableAFQMC.py: 使用 IterableDataset，以流式方式逐行读取数据，更适合处理无法完全加载内存的大型数据集。

AFQMCDataLoader.py: 负责将数据集打包成 DataLoader。它最关键的部分是 collate_fn 函数，这个函数定义了如何将一批文本样本转换为 BERT 模型能够接收的 input_ids, attention_mask 等张量格式。

BertForPairwiseCLS.py: 定义了用于句子对分类的模型。它在预训练的 BERT 模型之上增加了一个 Dropout 层和一个用于二元分类的线性层，通过取 BERT 输出的 [CLS] 向量来代表整个句子对的语义。

3. PyTorch 与 Transformer 基础 (test/)
这个目录包含了一系列用于学习和实验的基础脚本，涵盖了从 PyTorch 基础到 Transformer 核心组件的实现。

PytorchTest.py: 展示了 PyTorch 中基本的张量（Tensor）操作，例如创建、转换、维度计算以及如何使用 GPU (MPS) 进行运算。

SampleDataSet.py: 一个非常完整的 PyTorch 入门示例。它使用 FashionMNIST 数据集，详细展示了从数据加载、模型定义（一个简单的神经网络）、训练循环、测试循环，到模型保存和加载的完整流程。

CustomImageDataset.py & MyIterableDataSet.py: 提供了创建自定义数据集的两种模板。CustomImageDataset 适用于索引式的图片数据，而 MyIterableDataSet 适用于流式数据。

ModelTest.py: 集中展示了 Hugging Face transformers 库的核心用法，特别是 AutoTokenizer 和 AutoModel。内容包括：文本的分词、ID 转换、编码与解码，以及如何处理批次数据的填充（padding）和截断（truncation）。

TransfomerTest.py: 演示了 Hugging Face pipeline API 的强大功能，只需几行代码就能完成情感分析、文本生成、问答、摘要等常见 NLP 任务。

Single_Attention.py & Multi_Head_Attention.py: 这两个文件是理解 Transformer 模型核心的关键。它们从零开始，仅使用 PyTorch 实现了缩放点积注意力（Scaled Dot-Product Attention）和多头注意力（Multi-Head Attention）机制，并逐步构建出完整的 Transformer Encoder 层。
