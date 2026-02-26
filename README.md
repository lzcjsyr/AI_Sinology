# AI 汉学论文流水线 (CLI)

基于 `docs/` 设计清单实现的五阶段命令行流程：

1. 选题与构思（输出 `1_research_proposal.md`）
2. 史料搜集与交叉验证（输出 `2_final_corpus.yaml`）
3. 大纲构建（输出 `3_outline_matrix.yaml`）
4. 撰写初稿（输出 `4_first_draft.md`）
5. 修改与润色（输出 `5_final_manuscript.docx` 和 `5_revision_checklist.md`）

## 快速开始

1. 在 `core/config.py` 中按阶段配置模型与供应商（默认已给出示例）：

```python
PIPELINE_LLM_CONFIG = {
    "stage1": {"provider": "siliconflow", "model": "..."},
    "stage2_llm1": {"provider": "siliconflow", "model": "..."},
    "stage2_llm2": {"provider": "volcengine", "model": "doubao-seed-2-0-mini-260215"},
    "stage2_llm3": {"provider": "volcengine", "model": "..."},
    "stage3": {"provider": "siliconflow", "model": "..."},
    "stage4": {"provider": "siliconflow", "model": "..."},
    "stage5": {"provider": "siliconflow", "model": "..."},
}
```

默认 provider base URL 也写在 `core/config.py`：

```python
PROVIDER_DEFAULT_BASE_URLS = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
}
```

2. 配置 `.env`（只放 key，不放模型）：

```bash
SILICONFLOW_API_KEY=...
OPENROUTER_API_KEY=...
VOLCENGINE_API_KEY=...
```

3. 安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

4. 新建项目并跑完整流程（示例）：

```bash
python3 main.py \
  --new-project demo_ming_study \
  --idea "研究晚明通俗小说中的商人形象" \
  --scopes KR3j \
  --scope-dirs KR3j0160 \
  --max-fragments 8 \
  --stage2-sync-headroom 0.85 \
  --stage2-sync-max-ahead 128 \
  --yes
```

5. 继续已有项目：

```bash
python3 main.py --continue-project demo_ming_study
```

## 说明

- 阶段配置入口：`core/config.py`。可以清晰地按步骤切换 provider/model。
- 提示词入口：`prompts/*.yaml`。每个步骤一个文件，统一结构为：
  - `metadata.step`：步骤标识
  - `metadata.purpose`：步骤用途说明
  - `variables`：本步骤需要的输入变量说明
  - `expected_output`：模型返回格式说明
  - `prompt.system` / `prompt.user_template`：系统提示词与用户模板（变量缺失会直接报错终止）
- 阶段二可独立配置三套模型（`stage2_llm1/2/3`），并在同一处配置该模型的 `rpm/tpm`（见 `core/config.py` 的 `PIPELINE_LLM_CONFIG`）。
- 阶段二并发支持手工覆盖，也支持自动推导（推荐自动）：
  - `STAGE2_LLM1_CONCURRENCY` / `--stage2-llm1-concurrency`
  - `STAGE2_LLM2_CONCURRENCY` / `--stage2-llm2-concurrency`
- 当并发参数留空时，系统会根据该模型的 `rpm/tpm` 与请求 token 估算自动计算并发。
- 阶段二支持“同速并发”控制：`STAGE2_SYNC_HEADROOM`、`STAGE2_SYNC_MAX_AHEAD`（CLI 对应 `--stage2-sync-headroom`、`--stage2-sync-max-ahead`）。
- 阶段二仲裁支持并发：`STAGE2_ARBITRATION_CONCURRENCY` / `--stage2-arbitration-concurrency`。
- 阶段二支持按模型覆盖 RPM/TPM（默认：llm1=1000/100000，llm2=30000/5000000，llm3=1000/100000）：
  - `STAGE2_LLM1_RPM` / `STAGE2_LLM1_TPM`
  - `STAGE2_LLM2_RPM` / `STAGE2_LLM2_TPM`
  - `STAGE2_LLM3_RPM` / `STAGE2_LLM3_TPM`
- 阶段二检索范围来自 `data/kanripo_repos/KR-Catalog/KR/KR1.txt` 到 `KR4.txt` 的二级类目（如 `KR1a`、`KR3j`），CLI 展示格式为 `經部 [KR1a 易類]`。
- 阶段二支持双通道输入：交互式多选类目（方向键+Enter 勾选/取消，底部“开始”按钮确认）和手动目录输入（如 `KR1a0001`），两者会自动合并并去重。
- 交互式多选依赖 `prompt_toolkit`（已在 `requirements.txt` 中），若环境缺少依赖或非 TTY 终端，会自动降级为手动输入。
- 阶段二使用 LiteLLM 调用 OpenAI 兼容 API，并支持高并发筛选。
- 阶段二支持断点续传：`.cursor_llm1.json` 与 `.cursor_llm2.json`。
- `2_llm1_raw.jsonl` / `2_llm2_raw.jsonl` 是按主题展开后的行级结果。
  - 同一个 `piece_id` 会出现 N 次（N=目标主题数），这是“单次阅读、多主题判定”的展开结构，不是重复阅读。
- 阶段二日志统一写入 `2_stage_manifest.json`（包含所选 scopes、状态、重试信息和 `screening_audit`）。
- 产物按项目隔离存放在 `outputs/<project_name>/`。
