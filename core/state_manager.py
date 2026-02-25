from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.utils import ensure_dir, slugify


STAGE_FILES = {
    1: "1_research_proposal.md",
    2: "2_final_corpus.yaml",
    3: "3_outline_matrix.yaml",
    4: "4_first_draft.md",
    5: "5_final_manuscript.docx",
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

    def infer_state(self, project_name: str) -> ProjectState:
        project_dir = self.outputs_dir / project_name
        if not project_dir.exists():
            raise FileNotFoundError(f"项目不存在: {project_name}")

        if (project_dir / STAGE_FILES[5]).exists():
            next_stage = 6
        elif (project_dir / STAGE_FILES[4]).exists():
            next_stage = 5
        elif (project_dir / STAGE_FILES[3]).exists():
            next_stage = 4
        elif (project_dir / STAGE_FILES[2]).exists():
            next_stage = 3
        elif (project_dir / STAGE_FILES[1]).exists():
            next_stage = 2
        else:
            next_stage = 1

        return ProjectState(
            project_name=project_name,
            project_dir=project_dir,
            next_stage=next_stage,
            current_stage_name=STAGE_NAMES[next_stage],
        )

    @staticmethod
    def stage_name(stage_index: int) -> str:
        return STAGE_NAMES.get(stage_index, "未知阶段")
