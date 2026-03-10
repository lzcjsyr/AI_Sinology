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
    "stage2_llm1": {"provider": "aliyun", "model": "qwen3.5-flash"},
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
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
```

2. 配置 `.env`（只放 key，不放模型）：

```bash
SILICONFLOW_API_KEY=...
OPENROUTER_API_KEY=...
VOLCENGINE_API_KEY=...
ALIYUN_API_KEY=...

# 可选：同一 provider 多 key（逗号分隔），用于 LiteLLM Router 负载均衡
VOLCENGINE_API_KEYS=key_1,key_2
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
- 默认执行方式是“半自动”：
  - 每完成一个阶段，CLI 会回到项目进度列表并刷新状态
  - 只有你确认当前阶段成果后，才会继续进入下一阶段
  - `--yes` 仍会跳过这些确认提示，适合全自动批跑
- 提示词入口：`prompts/*.yaml`。每个步骤一个文件，统一结构为：
  - `metadata.step`：步骤标识
  - `metadata.purpose`：步骤用途说明
  - `variables`：本步骤需要的输入变量说明
  - `prompt.system` / `prompt.user_template`：系统提示词与用户模板（变量缺失会直接报错终止）
  - 阶段二当前拆分为 `stage2_screening`、`stage2_refinement`、`stage2_arbitration` 三个 prompt 步骤
- 阶段二可独立配置三套模型（`stage2_llm1/2/3`），并在同一处配置该模型的 `rpm/tpm`（见 `core/config.py` 的 `PIPELINE_LLM_CONFIG`）。
- 阶段二并发支持手工覆盖，也支持自动推导（推荐自动）：
  - `STAGE2_LLM1_CONCURRENCY` / `--stage2-llm1-concurrency`
  - `STAGE2_LLM2_CONCURRENCY` / `--stage2-llm2-concurrency`
- 阶段二粗筛批次支持独立调参：
  - `STAGE2_SCREENING_BATCH_MAX_CHARS` / `--stage2-screening-batch-max-chars`
- 当并发参数留空时，系统会根据该模型的 `rpm/tpm` 与请求 token 估算自动计算并发。
- 若配置了同一 provider 的多 key（如 `VOLCENGINE_API_KEYS`），系统会自动启用 LiteLLM Router 负载均衡，并按 key 数放大阶段二速率上限（`effective_rpm/tpm = rpm/tpm * key_count`）。
- 阶段二支持“同速并发”控制：`STAGE2_SYNC_HEADROOM`、`STAGE2_SYNC_MAX_AHEAD`（CLI 对应 `--stage2-sync-headroom`、`--stage2-sync-max-ahead`）。
- 阶段二仲裁支持并发：`STAGE2_ARBITRATION_CONCURRENCY` / `--stage2-arbitration-concurrency`。
- 阶段二支持按模型覆盖 RPM/TPM（默认：llm1=30000/10000000，llm2=30000/5000000，llm3=30000/5000000）：
  - `STAGE2_LLM1_RPM` / `STAGE2_LLM1_TPM`
  - `STAGE2_LLM2_RPM` / `STAGE2_LLM2_TPM`
  - `STAGE2_LLM3_RPM` / `STAGE2_LLM3_TPM`
- 阶段二检索范围来自 `data/kanripo_repos/KR-Catalog/KR/KR1.txt` 到 `KR4.txt` 的二级类目（如 `KR1a`、`KR3j`），CLI 展示格式为 `經部 [KR1a 易類]`。
- 阶段二开始时会先让你选择检索方式：
  - 按 `KR1a` 级别交互式勾选类目（方向键+Enter 勾选/取消，底部“开始”按钮确认）
  - 手动输入范围（支持混输 `KR1a`、`KR1a0001`、`KR2e0020` 等）
- 进入某种检索方式后，可随时返回上一层重新切换：
  - 交互式多选里按 `ESC` / `Ctrl+C` 或点“取消”可返回“检索方式”选择
  - 手动输入提示里输入 `b` 可返回上一层
- 手动输入会自动去重并做父子归并：例如 `KR1a,KR1a0006` 会合并为 `KR1a`。
- 如果手动输入的目录不存在，CLI 会明确提示不存在的目录名，并要求重新输入。
- 交互式多选依赖 `prompt_toolkit`（已在 `requirements.txt` 中），若环境缺少依赖或非 TTY 终端，会自动降级为手动输入。
- 阶段二使用 LiteLLM 调用 OpenAI 兼容 API，并支持高并发筛选。
- 阶段二会先基于 `kanripo_fragments.jsonl` 生成 `_processed_data/kanripo_screening_batches.jsonl`，按同一 `source_file` 内相邻片段做贪心合并后再进行粗筛。
- 阶段二支持断点续传：`_internal/stage2/.cursor_llm1.json` 与 `_internal/stage2/.cursor_llm2.json` 会按 batch 级索引恢复进度。
- `2_llm1_raw.jsonl` / `2_llm2_raw.jsonl` 是最终的片段级命中结果。
  - 仅对命中的 `piece_id + matched_theme` 落盘，避免为无关主题写入大规模负样本。
  - batch 粗筛命中后，会进入 `stage2_refinement` 精筛复核；每个 piece 会携带少量前后邻接上下文，但只允许判断当前正文。
  - 每条记录会附带 `screening_batch_id`、`localization_method`、`localization_bundle_id`、`reason`、`anchor_text` 等定位元数据；默认 `localization_method=piece_direct_with_neighbors`，定位范围是单 piece。
- 多次重试后仍失败的 `piece_id` 不再按“不相关”兜底，而会写入 `2_screening_failed_pieces.yaml`，并从自动仲裁链路中剔除，留待人工审核。
- 阶段二对外 YAML（`2_consensus_data.yaml`、`2_disputed_data.yaml`、`2_llm3_verified.yaml`、`2_final_corpus.yaml`）会自动做精简导出。
  - 文件顶部包含 `piece_count` 统计。
  - `records` 里只保留人工复核真正需要的字段，不再暴露 `screening_batch_id` 和各类 `localization_*` 遗留元数据。
- 阶段二日志统一写入 `_internal/stage2/2_stage_manifest.json`（包含所选 scopes、状态、重试信息和 `screening_audit`）。
- 阶段五采用逐段润色（默认按 `####` 小节切片），每完成一段立即回写 `5_final_manuscript.md`，并记录 `5_polish_progress.json` 以支持中断续跑。
- 产物按项目隔离存放在 `outputs/<project_name>/`。
