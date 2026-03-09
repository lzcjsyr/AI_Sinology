from __future__ import annotations

from pathlib import Path


def stage2_internal_dir(project_dir: Path) -> Path:
    return project_dir / "_internal" / "stage2"


def stage2_internal_path(project_dir: Path, filename: str) -> Path:
    return stage2_internal_dir(project_dir) / filename


def stage2_internal_json_path(project_dir: Path, filename: str) -> Path:
    return stage2_internal_path(project_dir, filename)


def resolve_stage2_json_path(project_dir: Path, filename: str) -> Path:
    internal_path = stage2_internal_json_path(project_dir, filename)
    if internal_path.exists():
        return internal_path

    legacy_path = project_dir / filename
    if legacy_path.exists():
        return legacy_path

    return internal_path


def resolve_stage2_internal_path(project_dir: Path, filename: str) -> Path:
    internal_path = stage2_internal_path(project_dir, filename)
    if internal_path.exists():
        return internal_path

    legacy_path = project_dir / filename
    if legacy_path.exists():
        return legacy_path

    return internal_path


def resolve_stage2_manifest_path(project_dir: Path) -> Path:
    return resolve_stage2_internal_path(project_dir, "2_stage_manifest.json")
