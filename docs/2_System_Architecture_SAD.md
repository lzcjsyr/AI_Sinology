# 系统架构设计文档 (System Architecture Document - SAD)

## 1. 技术栈声明

- **编程语言**：Python 3.10+
- **流程调度与提示词管理**：Python 工作流 + YAML 提示词模板（便于人工编辑与版本管理）
- **多模型 API 网关**：LiteLLM (用于统一封装大厂 API 调用，处理并发与自动路由)

## 2. 系统核心目录树结构

```text
thesis_agent_system/
├── main.py                          # 全局调度入口
├── .env                             # API Keys 等基础环境变量
├── core/                            # 核心公共组件
│   ├── config.py                    # 配置管理
│   ├── logger.py                    # 日志模块
│   ├── llm_client.py                # 跨模型并发调用封装层 (基于 LiteLLM)
│   ├── state_manager.py             # 项目状态推断与断点续传管理 (File-based State Machine)
│   └── utils.py                     # 通用工具箱（如 Markdown/YAML 的读写）
├── workflow/                        # 核心业务执行流 (分为5个阶段)
│   ├── stage1_topic_selection.py    # [阶段一] 选题顾问与构思 
│   ├── stage2_data_collection/      # [阶段二] 史料挖掘
│   │   ├── data_ingestion/          #  - 清洗各类原始数据源生成统一标准的结构片段
│   │   │   ├── parse_kanripo.py     # [核心主力] Kanripo 专属解析器 (纯代码数据转化，不牵涉LLM)
│   │   ├── archival_screening.py    # 双模型并发分片初筛 (向LLM发送阅读指令阶段)
│   │   └── archival_arbitration.py  # 史料争议仲裁与核心资产汇编
│   ├── stage3_outlining.py          # [阶段三] 大纲架构映射
│   ├── stage4_drafting.py           # [阶段四] 初稿起草 (RAG 限定)
│   └── stage5_polishing.py          # [阶段五] 定稿与排版 (输出严格符合学术规范的 .docx 论文)
├── prompts/                         # YAML/Jinja 提示词模板库，与 Python 剥离
├── outputs/                         # 依据按项目名的项目隔离运行成果目录
│   └── <project_name>/
│       ├── _processed_data/         # [由阶段2.1动态生成] 本项目专属的标准切片素材池
│       └── ...                      # 其他阶段产出物
└── data/                            
    └── kanripo_repos/               # [主力语料库] 外部未经清洗的原生经典文献资源池 (全局共享只读)
```

## 3. 各业务模块功能边界

- **`main.py`**：全局CLI交互入口。负责接收终端参数、渲染主菜单并分发任务流程。
- **`core/state_manager.py`**：项目状态机与断点续传模块。它负责扫描 `outputs/` 目录，通过文件探针推断项目进度，并在重跑时按阶段清理旧产物。
- **`core/llm_client.py`**：统一的 LiteLLM 调用封装层，负责同步/异步请求、响应结构归一化和 usage 提取。阶段二的并发与限流调度在 `workflow/stage2_data_collection/` 内实现。
- **`workflow/stage2_data_collection/data_ingestion/`**：数据萃取（Ingestion）组件。该组件负责纯代码的数据逻辑转化工作，绝对不触碰任何 LLM 逻辑。其主要任务是定向提取指定的底层原始文档，进行字符串清洗与微观切割，从而生成带有极细 `piece_id` 的标准素材池。
  - **`parse_kanripo.py`**：【核心生肉语料基底】针对 `data/kanripo_repos` 的专属解析器。Kanripo 文本是本系统史料挖掘的绝对核心。提取规则为：读取指定范围内的所有子文件夹和 `.txt` 文件，剔除文件头部的非史料元数据标记（如果有）。切分时必须极其微观，代码需主动识别并利用原文件内置的形如 `<pb:XXX>` 的页码/语义标签进行切片。生成的片段其 `piece_id` 应当提取该标签内部的字符串（即去除尖括号，如 `pb:KR1a0001_tls_001-1a`），从而精确定位到该史料对应的古籍原书书名、版本、卷号及页码。这些碎片被统一定向输出至项目专属的 `outputs/<project_name>/_processed_data/kanripo_fragments.jsonl`，确保大模型每次读取和引用的都是这种极细且出处可靠的古文，且严格保证多项目并发时的物理隔离。
- **`workflow/stage2_data_collection/archival_screening.py`**：本系统“干重活”的引擎模块。此模块处于数据被 Ingestion 组件切分完毕后的环节。它将直接读取生成的项目专属 `outputs/<project_name>/_processed_data/` 碎片池，并异步向双侧大模型派发穷举阅读指令。收集返回结果后记录 `is_relevant` 等 JSON 数据。此阶段不再负责底层文献的菜单展示或文本解析（那是前置和Ingestion的职责）。

## 4. 全局交互与项目状态流转 (CLI Orchestration & State Management)

系统采用单向数据流和文件系统作为状态机的持久化层（File-based State Machine）。**`main.py` 负责渲染 UI 表现层，而底层的核心状态则完全委托给封装的 `core/state_manager.py` 模块处理**，统筹以下交互与断点续传逻辑：

### 4.1 项目初始化与扫描流

1. **启动应用**：用户在终端输入 `python main.py` 启动程序。
2. **模式选择**：系统拦截指令并打印交互菜单：
   - `[1] 创建新研究项目`
   - `[2] 继续现有项目`
3. **现有项目扫描**：若用户选择 `[2]`，系统自动扫描 `outputs/` 目录下的所有子文件夹。
4. **状态推断 (State Inference)**：通过探查各子文件夹内的文件清单，反推当前项目处于第几阶段。例如：
   - 仅存在 `1_research_proposal.md` -> 系统提示：“当前处于【阶段二：史料搜集】，是否继续向下执行？”
   - 存在 `2_final_corpus.yaml` 或 `2_final_corpus.json` -> 系统提示：“当前处于【阶段三：大纲构建与逻辑推演】，是否继续？”
   - 存在 `3_outline_matrix.yaml` -> 系统提示：“当前处于【阶段四：撰写初稿】，是否继续？”
   - 存在 `4_first_draft.md` -> 系统提示：“已进入最终打磨环节【阶段五：修改与润色】，即将生成排版 Docx。”

### 4.2 中断点与容灾回滚机制 (Checkpoint & Recovery)

针对最耗时、最容易网络中断的阶段 2.2（双模型大并发 API 阅读）：

- 所有生成的数据一律采用追加模式（`a+`）分别写入 `2_llm1_raw.jsonl` 与 `2_llm2_raw.jsonl`。考虑到双侧大模型（如 DeepSeek 和 GPT-4o）的 API 响应速度和限流策略不同，处理进度注定会产生错位，系统必须为并发线程维护**双游标卡**（即 `.cursor_llm1.json` 与 `.cursor_llm2.json`），独立记录各自成功写入的最后一条 `piece_id`。
- 一旦用户通过 `Ctrl+C` 或因网络熔断导致程序崩溃，下次通过主控复位现有项目并在阶段二重新启动时，系统必须分别读取这两个游标，各自从剩余未读切片处满血复活进行断点续传（**Resume from Breakpoint**），彻底杜绝数据复写或丢失。

这种通过底层文件探针（探测当前项目下生成了哪些核心产物文件名）来实现的粗粒度状态机，结合细粒度的游标卡片，使系统呈现出无状态交互和零学习成本的 CLI 极客体验。
