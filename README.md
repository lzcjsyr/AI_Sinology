# AI 汉学论文流水线 (CLI)

基于 `docs/` 设计清单实现的五阶段命令行流程：

1. 选题与构思（输出 `1_research_proposal.md`）
2. 史料搜集与交叉验证（输出 `2_final_corpus.yaml`）
3. 大纲构建（输出 `3_outline_matrix.json`）
4. 撰写初稿（输出 `4_first_draft.md`）
5. 修改与润色（输出 `5_final_manuscript.docx` 和 `5_revision_checklist.md`）

## 快速开始

1. 配置环境变量（或 `.env`）：

```bash
API_KEY=...
BASE_URL=...
MODEL=...
```

2. 新建项目并跑完整流程（示例）：

```bash
python3 main.py \
  --new-project demo_ming_study \
  --idea "研究晚明通俗小说中的商人形象" \
  --scopes KR3j0160 \
  --max-fragments 8 \
  --yes
```

3. 继续已有项目：

```bash
python3 main.py --continue-project demo_ming_study
```

## 说明

- 阶段二支持断点续传：`.cursor_llm1.json` 与 `.cursor_llm2.json`。
- 产物按项目隔离存放在 `outputs/<project_name>/`。
- 代码仅使用 Python 标准库，无额外依赖。
