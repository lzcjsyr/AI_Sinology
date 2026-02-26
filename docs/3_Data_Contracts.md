# 数据契约与接口规范 (Data Contracts & Interface Specs)

在本项目由五个业务阶段（对应多个核心脚本）驱动的流水线中，上游步骤的输出是下游步骤的强约束输入。所有的中间产物（包括数据切片）均被严格沙箱化，默认保存在项目专属目录 `outputs/<project_name>/` 中。为了确保多智能体（Multi-Agent）和不同业务脚本之间能够无缝协作，必须严格遵守以下数据结构和格式规范。

---

## 阶段一：选题与构思 (Topic Selection & Conception)

### [输入] 用户原始需求 / 预设 Prompt

- **信息来源**：人类用户的模糊研究意向（如“研究晚明通俗小说中的商人形象”）。
- **形式**：由 `main.py` 通过 CLI 接收的字符串参数，或存放在 `.env` / `prompts/` 目录下的初始化提示词集。

---

### [输出] 1_research_proposal.md

本阶段产出采用带有 YAML Front Matter 头的 Markdown (`.md`) 格式，以兼顾代码解析与人类阅读。

- **生成方式**：按以下六个部分顺序，分六次调用 LLM 串行生成。其中，“目标主题列表”（第6部分的核心产出）由脚本提取并格式化为文件顶部的 YAML 区块，其余部分作为正文。
- **文件结构要求**：

**1. YAML Front Matter (系统配置区)**：
必须位于文件最开头，被 `---` 包裹。下游的 Python 脚本将**仅解析此区域**。

```yaml
---
target_themes:
  - theme: "祈雨"
    description: "包括皇帝和民间百姓通过宗教活动和民间习俗请求降雨..."
  - theme: "商人形象"
    description: "晚明通俗小说中的商贾、市井闲人等商业群体代表..."
---
```

> [!IMPORTANT]
> `target_themes` 是下游**阶段二的唯一输入基准**。后续主题比对必须与此处键值严格一致 (Exact Match)。

**2. 正文区 (Markdown)**：
在 `---` 之后，呈现为标准的学术文本：

1. **研究背景与问题陈述 (Problem Statement)**：约 500 字，界定核心现象与语境。
2. **核心研究问题 (Research Questions)**：约 500 字，提出 3-5 个主干及子问题。
3. **学术史述评 (Literature Review)**：约 2000 字，对前人研究进行主题梳理与评价。
4. **研究思路与切入点 (Methodology & Perspective)**：约 1000 字，明确理论工具与初步大纲。
5. **预期创新与学术价值 (Significance & Innovation)**：约 500 字，说明理论建构或史料发掘贡献。
6. **史料检索策略说明 (Archival Strategy)**：对目标古籍库和文献范围的人类可读界定（机器不解析）。

---

## 阶段 2.1：异构数据接入层 (Data Ingestion)

### [输入] 用户指令与原始语料集合

- **检索范围参数 (Scope Selection)**：生成前，系统通过 CLI 收集用户的确切界定范围。操作 Kanripo 语料时，程序会首先解析 `data/kanripo_repos/KR-Catalog/KR/KR?.txt` 等索引文件，并在终端生成一个直观的多选项列表（例如显示 `[ ] KR1a 易類` 供用户直接上下移动勾选）。或者，为了照顾更微观的需求，CLI 依然允许用户直接以字符串形式精确输入一部书的具体文件夹名称（例如 `KR1a0001`）。解析器（Ingestion层）将根据此匹配规则提取对应 `.txt` 文件，极大节省计算资源。
- **原生史料库**：存放在 `data/kanripo_repos/` 等类似目录下的古籍原始文本文件（目前以 Kanripo 作为绝对主力数据源，包含完整的版本及页码标识等元数据）。

---

### [输出] 统一标准史料片段池

- **数据提取与清洗规则（以 Kanripo 为例）**：提取脚本读取设定范围内的 `.txt` 文件，清洗并剔除古籍头部的元数据或校阅标记。
  - **书名元数据提取**：在丢弃头部元数据前，脚本必须首先扫描并提取形如 `#+TITLE: 周易鄭康成注` 的标签内容，将其作为该文件内所有生成片段的 `source_file` 字段值。
  - **坐标切片**：随后提取脚本需依据文本内嵌的 `<pb:XXX>` 等翻页分隔符对内容进行切片，并提取标签内部的字符串作为 `piece_id`（即去除尖括号，如提取出 `pb:KR1a0001_tls_001-1a` 以作为 ID）。这直接代表了包含书名、版本、卷次和页码的绝对学术定位。
- **文件名格式**：`outputs/<project_name>/_processed_data/<corpus_name>_fragments.jsonl`（例如基于 Kanripo 语料库生成 `kanripo_fragments.jsonl`。**注意**：此文件本质上仅为阶段性数据中转池。为了节省磁盘空间，建议在最终排版完成后，该文件及其同类的临时分片应当被脚本清理抛弃，或由用户定期清除）。
- **JSON Schema (单行定义)**：不涉及任何 LLM 分析，纯代码提取。

```json
{
  "piece_id": "string",       // 必填。依据原生语义标签截取的唯一坐标，如 "pb:KR1a0001_tls_001-1a"
  "source_file": "string",    // 必填。解析原生文件头部 #+TITLE: 标签所得的确切书名，如 "周易鄭康成注"
  "original_text": "string"   // 必填。清洗干净、长短均等的纯文本片段原文
}
```

---

## 阶段 2.2：双模型独立审核与比对分流 (Independent Review & Shunting)

### [输入] 目标主题列表与标准片段池

- **业务输入限制**：第一阶段产出的 `1_research_proposal.md` 中，**只能由脚本提取其顶部的 YAML Front Matter 区块内的 `target_themes` 数组**，作为当前并发判断的基准系统提示词（System Prompt）。《研究计划书》的其余论证正文对本评估阶段不可见。
- **数据输入**：前置打散的 `outputs/<project_name>/_processed_data/` 碎片化 JSONL 语料池。

---

### [输出] 独立分析原始档案

此阶段的输出组合分为两部分构成，以达到极致的 Token 节省并兼顾下游使用效率：

**1. 大模型 (LLM) 结构化输出部分**：
考虑到一段史料需要针对多个目标主题进行判定，LLM 被要求返回一个**包含所有预定义主题判定结果的数组列（Array）**。
LLM 的 JSON 必须极致精简，**严格禁止**其复述原文或返回 `piece_id` 等已知信息。为了节省 Token，**无论是否相关，`is_relevant` 字段必须真实输出**。但如果判定为不相关（`is_relevant: false`），则**无需输出理由，直接省去或置空 `reason` 和 `relevance_level` 字段**：

```json
{
  "matches": [
    {
      "theme": "string",              // 需判定的目标主题名
      "is_relevant": boolean,         // 必须输出。相关为 true，不相关为 false
      "reason": "string | null",      // 说明理由。若 is_relevant 为 false，此字段可为 null 或省略
      "relevance_level": "string | null"  // 若 is_relevant 为 false，此字段可为 null 或省略
    }
  ]
}
```

**2. 落盘文件 (`2_llm1_raw.jsonl` 和 `2_llm2_raw.jsonl`) 的完整单行 Schema**：
调度的纯 Python 脚本（`archival_screening.py`）在接收到 LLM 数组后，**以该数组内的每一个 `theme` 为基本单位**做对象扁平铺开。
脚本在内存中直接将其与上游生成的 `piece_id`、`source_file` 和 `original_text` 缝合。如果某个主题的判定为不相关，依然会生成一条记录（记录其 `is_relevant: false`），以供下游比对：

```json
{
  "piece_id": "string",           // 代码直接注入
  "source_file": "string",        // 代码直接注入
  "original_text": "string",      // 代码直接注入
  "matched_theme": "string",      // 拆解自 LLM 返回的 theme
  "is_relevant": boolean,         // 拆解自 LLM 返回的 boolean (true/false)
  "reason": "string | null",      // 拆解自 LLM 返回的理由或 null
  "relevance_level": "string | null" // 拆解自 LLM 返回的评级或 null
}
```

> [!TIP]
> **Token 开销与速度优化：** 将大模型输出字段与最终落盘文件 Schema 解耦，大模型一次只需要输出几个极短的判定字段。对于不相关的主题，通过省略 `reason` 字段完美规避了长文本生成的 Token 开销。这成倍加快了生成速度，并且使得下游（阶段2.3）的文件由于天然带有 `is_relevant` 字段，可直接依赖其进行清晰严密的布尔判断和差集比对。

## 阶段 2.3：共识与争议分流 (Consensus & Dispute Shunting)

### [输入] 双侧独立初核结果

- **底层数据源**：阶段 2.2 产出的 `2_llm1_raw.jsonl` 与 `2_llm2_raw.jsonl` 双份独立核查报告。

---

### [输出] 共识与争议分流档

此环节由纯代码脚本对前两个 JSONL 文件进行精准比对生成，不涉及任何 LLM 逻辑。格式由 JSONL 转换为人类友好的 YAML，便于人工抽检或供后续第三方智能体仲裁。

- **文件名**：`2_consensus_data.yaml` (无分歧档案) 和 `2_disputed_data.yaml` (存在分歧的档案)

**无分歧档案 `2_consensus_data.yaml` 数据结构**：
完全继承单条原格式，只保留双盲判定一致的档案。
**判定共识（Consensus）的标准（细化至“Theme”维度）**：

- **明确共识**：基于同一个 `piece_id`，只要 LLM1 与 LLM2 的记录中**均包含同一个 `theme` 且 `is_relevant` 均为 `true`**，即在该主题上达成共识。共识的获取**不考虑**相关等级（`relevance_level`）或理由差异，只要“主题”和“是否相关”两个字段匹配即可写入共识档案。
- **彻底无关**：基于同一个 `piece_id` 和特定的 `theme`，若双侧模型的记录中 **`is_relevant` 均为 `false`**，即视该史料片段在该主题下无价值，直接丢弃（不写入任何档案）。若该片段针对所有目标主题的双侧判定均为 `false`，则该史料片段被彻底抛弃。

```yaml
- piece_id: "pb:KR1a0001_tls_001-1a"
  source_file: "周易"
  original_text: |
    是日，商贾云集，市井繁华...
  matched_theme: "商人形象"
  is_relevant: true
  reason: "明确描写了市井商人的聚集场景。"
  relevance_level: "HIGH"
```

**存在分歧的档案 `2_disputed_data.yaml` 数据结构**：
**判定争议（Dispute）的唯一标准（基于主题差集）**：针对同一段史料和特定的 `theme`，**只有一个模型判定其为相关（`is_relevant: true`），而另一个模型明确判定为不相关（`is_relevant: false`）**，才会需要仲裁。
（例如：如果两者都在“祈雨”上达成共识，但在“商人形象”上只有一方认为相关，另一方返回了 false，那么共识池里写“祈雨”，争议池里仅仅针对“商人形象”单独启动一轮仲裁判定。）
为保证第三个 LLM 能够充分比较前两者的判断差异，此文档的嵌套结构必须清楚展现争议双方。**注意：为极致节省 Token 并保持逻辑一致，阶段 2.2 设定了判定为不相关时可不输出理由。因此，此处的 `is_relevant: false` 和 `reason: null` 是完全忠实于上游 LLM 实际产生的判断记录，依靠真实的 `is_relevant` 字段驱动后续比对工作。**

```yaml
- piece_id: "pb:KR1a0001_tls_001-2a"
  source_file: "周易"
  original_text: |
    祈雨之时，一聚散游手无赖之徒...
  matched_theme: "商人形象"       # 该主题只有一方认为相关
  llm1_result: 
    is_relevant: true
    relevance_level: "LOW"
    reason: "提到游手好闲聚众人员，也许暗含着市井人员的初步集结。"
  llm2_result:
    is_relevant: false
    relevance_level: null
    reason: null                  # 忠实于 LLM2 在阶段 2.2 时输出的无理由不相关判定
```

```yaml
- piece_id: "pb:KR1a0001_tls_001-2a"
  source_file: "周易"
  original_text: |
    大旱，王乃使祈雨...
  matched_theme: "祈雨"
  llm1_result: 
    is_relevant: true
    relevance_level: "HIGH"
    reason: "包含明确的祈雨情况与求雨目的。"
  llm2_result:
    is_relevant: false
    relevance_level: null
    reason: null                  # 忠实于 LLM 输出的无理由不相关判定
```

---

## 阶段 2.4：第三方复核与最终总库汇编 (Arbitration & Compilation)

### [输入] 争议分流档与无争议底稿

- **需仲裁输入**：2.3 阶段产生的 `2_disputed_data.yaml`（存在分歧的档案），供第三方大模型 (LLM3) 进行重裁。
- **无需仲裁输入**：2.3 阶段产生的 `2_consensus_data.yaml`（无分歧共识档案），作为最终合并的基石底稿。

---

### [输出] 2_llm3_verified.yaml (仲裁结果)

由第三方大模型读取争议档案并做出最终独立判定。若相关，则提取保留；若无关，则直接丢弃。

- **格式要求**：必须剥离 2.3 阶段争议文件中的双模型嵌套对比结构，打平恢复为最扁平的标准结构，即与最初阶段 2.2 `raw.jsonl` 单条提取的字段要素**完全一致**（转为 YAML 展现）。这保证了下一步合并系统的无缝对接。

---

### [输出] 2_final_corpus.yaml / 2_final_corpus.json (核心总库)

**明确定义**：`2_final_corpus.yaml` 在物理和逻辑上，就是 `2_consensus_data.yaml`（无分歧共识档案）与 `2_llm3_verified.yaml`（存在争议但经过第三方大模型重裁后的有效档案）这**两个文件的直接全量合并**，作为最后沉淀出的核心 RAG 底料；系统会同时落盘等价内容到 `2_final_corpus.json` 供后续阶段直接读取。

- **数据结构与契约**：使用 YAML 格式，保留多行文本 (`|` 语法)。由于 `2_llm3_verified.yaml` 已经被强制要求剥离嵌套争议，格式恢复为最初单源分析时的扁平结构，因此合并拼接后的核心史料总库格式要素与最初阶段**完全一致**。

```yaml
- piece_id: "pb:KR1a0001_tls_001-1a"
  source_file: "周易"
  original_text: |
    是日，商贾云集，市井繁华...
  matched_theme: "商人形象"
  is_relevant: true
  reason: |
    明确描写了市井商人的聚集场景，能够佐证晚明城市经济的繁荣，符合商人形象的主题要求。
  relevance_level: "HIGH"
```

---

## 阶段三：大纲构建与逻辑推演 (Outlining)

### [输入] 宏观计划书与微观史料库

- **宏观纲领**：第一阶段产出的 `1_research_proposal.md` 全文，提供核心问题、文献回顾与预期学术创新方向。
- **底层支撑**：第二阶段收口输出的 `2_final_corpus.json`（与 `2_final_corpus.yaml` 等价），提供全部确定相关且评级可靠的实体史料阵列。

---

### [输出] 3_outline_matrix.yaml

系统将原始需求中的大纲文档结构化为 YAML 契约，以彻底锁死“论点与史料（piece_id）的映射关系”，拒绝空泛论证。此文件也是阶段四撰写初稿的导航图。

- **核心契约要求**：最底层的 `evidence_anchors` 必须仅能包含 `2_final_corpus.json` 中客观存在的合法 `piece_id`，不允许虚构不存在的 ID。
- **结构示意**：下方使用 JSON 仅用于展示字段层级，实际落盘文件为 YAML。

```json
{
  "thesis_statement": "用一句凝练的话概括全篇的核心结论与主旨...",
  "chapters": [
    {
      "chapter_title": "第一章 晚明市镇的繁华与社会生态",
      "chapter_argument": "说明本章的核心分论点及其在论证长链中的位置...",
      "sections": [
        {
          "section_title": "第一节 商贾云集的文本表现",
          "section_transition": "本节的承上启下与总体逻辑引言...",
          "sub_sections": [
            {
              "sub_section_title": "一、 市井空间的文本建构",
              "sub_section_argument": "本小节的微观推演逻辑...",
              "evidence_anchors": [
                "pb:KR1a0001_tls_001-1a",
                "pb:KR1a0001_tls_001-2a"
              ]
            }
          ],
          "counter_arguments_rebuttals": "预判可能的学术质疑并设下回应方案..."
        }
      ]
    }
  ]
}
```

---

## 阶段四：撰写初稿 (Drafting)

### [输入] YAML 论纲契约与 JSON 史料库左连接 (Data JOIN)

- **逻辑图纸**：第三阶段产出的 `3_outline_matrix.yaml`（骨干与 `piece_id` 的绑定树）。
- **内容抽屉**：第二阶段产出的 `2_final_corpus.json`（根据 `piece_id` 提取对应的 `original_text`）。
- **装配要求**：该阶段执行脚本必须在内存中，提取 YAML 里的 `evidence_anchors` 数组，并在 JSON 史料字典中完成 JOIN（左连接）操作。将原始史料文本与对应的大纲逻辑节点拼装绑定后，一并送入 Writer Agent 进行生成，以此彻底断绝大模型的史料幻觉。

---

### [输出] 4_first_draft.md

接收阶段三的 YAML 大纲和阶段二的史料 JSON，运行带有检索增强生成（RAG）限制的写手模型，输出学术论文初稿文本。

- **文件格式**：标准的学术长文本格式 (`.md`)
- **生成结构强制要求**：

  1. **绪论 (Introduction)**：展开提问、学术史述评与中心论点。
  2. **本论部分 (Body Paragraphs)**：每个段落的微观推演模式**必须**严格固化为：
     - **主题句 (Topic Sentence)** 引领段落逻辑。
     - **呈现权威史料**（必须精准复制 `original_text` 并不做任何篡改删短）。
     - **核心阐释分析**（基于史料推演）。
     - **段落小结**。
  3. **结论 (Conclusion)**：研究总结及局限性说明。
  4. **初排版引注区 (Draft Citations)**：文内暂存的出处注释。

---

## 阶段五：修改与润色 (Revision & Polishing)

### [输入] 初次排版稿件及目标期刊标准

- **待审原件**：第四阶段生成的全文 `4_first_draft.md`。
- **防篡改对照组**：包含原文和出处的 `2_final_corpus.json`，用于复查 Writer Agent 是否在行文中私自篡改了原始史料字符。
- **样式指导**：目标期刊或学位论文的特定排体例要求指令（如“请按国标 GB/T 7714-2015 格式化所有文献引用”）。

---

### [输出] 5_final_manuscript.docx

经格式校验与语体润色后交付给用户的、符合学术规范且经过严格排版优化的最终 `.docx` 格式学术主文档。

- **强制包含内容**：
  1. 标准化中英文摘要与关键词。
  2. 语言洗练、严守学术规范的整篇正文（脚注编号对应）。
  3. 按 GB/T 7714 或同等国标统一生成排版的参考文献表。

### 5_revision_checklist.md

伴随主文档一并生成的机器自检日志。

- **强制包含内容**：
  1. 逻辑修改溯源与盲点排查记录。
  2. 引文原始出处二次抽检复核日志（例如：“对 135 条引文与原始文本进行比对，确认文字零增删零篡改率 100%”）。
  3. 学术规范格式匹配自测报告。
