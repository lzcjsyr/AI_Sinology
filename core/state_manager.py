from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.utils import ensure_dir, read_json, slugify


STAGE_STATUS_NOT_STARTED = "not_started"
STAGE_STATUS_IN_PROGRESS = "in_progress"
STAGE_STATUS_COMPLETED = "completed"

STAGE_STATUS_LABELS = {
    STAGE_STATUS_NOT_STARTED: "未开始",
    STAGE_STATUS_IN_PROGRESS: "进行中",
    STAGE_STATUS_COMPLETED: "已完成",
}

STAGE_COMPLETION_ARTIFACTS = {
    1: ["1_research_proposal_meta.json"],
    2: ["2_final_corpus.json", "2_final_corpus.yaml"],
    3: ["3_outline_matrix.yaml"],
    4: ["4_first_draft.md"],
    5: ["5_final_manuscript.docx"],
}

STAGE_IN_PROGRESS_ARTIFACTS = {
    1: ["1_research_proposal.md"],
    2: [
        "2_stage_manifest.json",
        "_processed_data/kanripo_fragments.jsonl",
        "2_llm1_raw.jsonl",
        "2_llm2_raw.jsonl",
        ".cursor_llm1.json",
        ".cursor_llm2.json",
        "2_stage_failure_report.md",
    ],
    3: ["3_outline_matrix.json"],
    4: [],
    5: ["5_final_manuscript.md", "5_revision_checklist.md"],
}

STAGE_RESET_ARTIFACTS = {
    1: ["1_research_proposal.md", "1_research_proposal_meta.json"],
    2: [
        "_processed_data/kanripo_fragments.jsonl",
        "2_stage_manifest.json",
        "2_llm1_raw.jsonl",
        "2_llm2_raw.jsonl",
        ".cursor_llm1.json",
        ".cursor_llm2.json",
        "2_consensus_data.yaml",
        "2_consensus_data.json",
        "2_disputed_data.yaml",
        "2_disputed_data.json",
        "2_llm3_verified.yaml",
        "2_llm3_verified.json",
        "2_final_corpus.yaml",
        "2_final_corpus.json",
        "2_stage_failure_report.md",
    ],
    3: ["3_outline_matrix.yaml", "3_outline_matrix.json"],
    4: ["4_first_draft.md"],
    5: ["5_final_manuscript.md", "5_revision_checklist.md", "5_final_manuscript.docx"],
}

STAGE_NAMES = {
    1: "阶段一：选题与构思",
    2: "阶段二：史料搜集与交叉验证",
    3: "阶段三：大纲构建与逻辑推演",
    4: "阶段四：撰写初稿",
    5: "阶段五：修改与润色",
    6: "全部完成",
}


@dataclass
class ProjectState:
    project_name: str
    project_dir: Path
    next_stage: int
    current_stage_name: str


@dataclass(frozen=True)
class StageProgress:
    stage_index: int
    stage_name: str
    status: str

    @property
    def status_label(self) -> str:
        return STAGE_STATUS_LABELS.get(self.status, "未知状态")


class StateManager:
    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        ensure_dir(outputs_dir)

    def list_projects(self) -> list[str]:
        if not self.outputs_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.outputs_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def create_project(self, project_name: str) -> ProjectState:
        safe_name = slugify(project_name)
        project_dir = self.outputs_dir / safe_name
        ensure_dir(project_dir)
        ensure_dir(project_dir / "_processed_data")
        return ProjectState(
            project_name=safe_name,
            project_dir=project_dir,
            next_stage=1,
            current_stage_name=STAGE_NAMES[1],
        )

    def get_project(self, project_name: str) -> ProjectState:
        project_dir = self.outputs_dir / project_name
        if not project_dir.exists() or not project_dir.is_dir():
            raise FileNotFoundError(f"项目不存在: {project_name}")
        return self.infer_state(project_name)

    @staticmethod
    def _exists_any(project_dir: Path, relative_paths: list[str]) -> bool:
        return any((project_dir / relative_path).exists() for relative_path in relative_paths)

    @staticmethod
    def _read_stage2_manifest_status(project_dir: Path) -> str:
        manifest_path = project_dir / "2_stage_manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            payload = read_json(manifest_path)
        except Exception:  # noqa: BLE001
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("status") or "").strip()

    def _stage_status(self, project_dir: Path, stage_index: int) -> str:
        completion_files = STAGE_COMPLETION_ARTIFACTS.get(stage_index, [])
        if self._exists_any(project_dir, completion_files):
            return STAGE_STATUS_COMPLETED

        if stage_index == 2:
            manifest_status = self._read_stage2_manifest_status(project_dir)
            if manifest_status:
                return STAGE_STATUS_IN_PROGRESS

        in_progress_files = STAGE_IN_PROGRESS_ARTIFACTS.get(stage_index, [])
        if self._exists_any(project_dir, in_progress_files):
            return STAGE_STATUS_IN_PROGRESS

        return STAGE_STATUS_NOT_STARTED

    @staticmethod
    def suggest_next_stage(progress: list[StageProgress]) -> int:
        for item in progress:
            if item.status == STAGE_STATUS_IN_PROGRESS:
                return item.stage_index
        for item in progress:
            if item.status == STAGE_STATUS_NOT_STARTED:
                return item.stage_index
        return 6

    @staticmethod
    def highest_completed_stage(progress: list[StageProgress]) -> int:
        completed = [item.stage_index for item in progress if item.status == STAGE_STATUS_COMPLETED]
        return max(completed) if completed else 0

    def infer_stage_progress(self, project_name: str) -> list[StageProgress]:
        project_dir = self.outputs_dir / project_name
        if not project_dir.exists():
            raise FileNotFoundError(f"项目不存在: {project_name}")

        statuses = [self._stage_status(project_dir, stage_index) for stage_index in range(1, 6)]

        # If a downstream stage is completed, upstream stages are treated as completed too.
        for idx, status in enumerate(statuses):
            if status != STAGE_STATUS_COMPLETED:
                continue
            for prev_idx in range(0, idx):
                statuses[prev_idx] = STAGE_STATUS_COMPLETED

        return [
            StageProgress(
                stage_index=stage_index,
                stage_name=STAGE_NAMES[stage_index],
                status=statuses[stage_index - 1],
            )
            for stage_index in range(1, 6)
        ]

    def collect_artifacts_from_stage(self, project_dir: Path, start_stage: int) -> list[Path]:
        artifacts: list[Path] = []
        for stage_index in range(max(1, start_stage), 6):
            for relative_path in STAGE_RESET_ARTIFACTS.get(stage_index, []):
                full_path = project_dir / relative_path
                if full_path.exists():
                    artifacts.append(full_path)
        return artifacts

    def clear_artifacts_from_stage(self, project_dir: Path, start_stage: int) -> list[Path]:
        removed: list[Path] = []
        for artifact_path in self.collect_artifacts_from_stage(project_dir, start_stage):
            if artifact_path.exists():
                artifact_path.unlink()
                removed.append(artifact_path)
        return removed

    def infer_state(self, project_name: str) -> ProjectState:
        project_dir = self.outputs_dir / project_name
        if not project_dir.exists():
            raise FileNotFoundError(f"项目不存在: {project_name}")

        progress = self.infer_stage_progress(project_name)
        next_stage = self.suggest_next_stage(progress)

        return ProjectState(
            project_name=project_name,
            project_dir=project_dir,
            next_stage=next_stage,
            current_stage_name=STAGE_NAMES[next_stage],
        )

    @staticmethod
    def stage_name(stage_index: int) -> str:
        return STAGE_NAMES.get(stage_index, "未知阶段")
