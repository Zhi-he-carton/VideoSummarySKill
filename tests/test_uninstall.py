from __future__ import annotations

from pathlib import Path

from scripts import uninstall


def make_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "video-summary"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: video-summary\n---\n", encoding="utf-8"
    )
    (skill_dir / "pyproject.toml").write_text(
        "[project]\nname='video-summary'\n", encoding="utf-8"
    )
    for name in uninstall.RUNTIME_DIRECTORIES:
        runtime = skill_dir / name
        runtime.mkdir()
        (runtime / "dependency.bin").write_bytes(b"dependency")
    return skill_dir


def remove_test_cache(path: Path) -> None:
    uninstall._remove_path(path)


def test_preview_lists_owned_runtime_without_removing_it(tmp_path: Path) -> None:
    skill_dir = make_skill(tmp_path)

    result = uninstall.uninstall(skill_dir, dry_run=True)

    assert result["ok"] is True
    assert result["status"] == "preview"
    assert all(item["present"] for item in result["targets"])
    assert all((skill_dir / name).exists() for name in uninstall.RUNTIME_DIRECTORIES)


def test_confirmed_uninstall_removes_runtime_but_preserves_sources(
    tmp_path: Path,
) -> None:
    skill_dir = make_skill(tmp_path)

    result = uninstall.uninstall(
        skill_dir,
        dry_run=False,
        executable=tmp_path / "python",
        cache_cleaner=remove_test_cache,
    )

    assert result["ok"] is True
    assert result["status"] == "uninstalled"
    assert all(item["removed"] for item in result["targets"])
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "pyproject.toml").is_file()


def test_model_cleanup_is_opt_in_and_keeps_unrelated_cache(tmp_path: Path) -> None:
    skill_dir = make_skill(tmp_path)
    model_root = tmp_path / "huggingface" / "hub"
    target_models = [
        model_root / ("models--" + repository.replace("/", "--"))
        for repository in uninstall.MODEL_REPOSITORIES
    ]
    unrelated = model_root / "models--someone--unrelated"
    for path in [*target_models, unrelated]:
        path.mkdir(parents=True)
        (path / "weights.bin").write_bytes(b"weights")

    result = uninstall.uninstall(
        skill_dir,
        include_models=True,
        dry_run=False,
        model_cache_root=model_root,
        executable=tmp_path / "python",
        cache_cleaner=remove_test_cache,
    )

    assert result["ok"] is True
    assert all(not path.exists() for path in target_models)
    assert unrelated.is_dir()


def test_uninstaller_refuses_to_remove_its_active_runtime(tmp_path: Path) -> None:
    skill_dir = make_skill(tmp_path)
    active_python = skill_dir / ".venv" / "bin" / "python"
    active_python.parent.mkdir()
    active_python.write_bytes(b"")

    result = uninstall.uninstall(skill_dir, dry_run=False, executable=active_python)

    assert result["ok"] is False
    assert result["error"]["code"] == "active_runtime"
    assert (skill_dir / ".venv").is_dir()


def test_invalid_skill_directory_is_never_removed(tmp_path: Path) -> None:
    candidate = tmp_path / "not-a-skill"
    candidate.mkdir()
    marker = candidate / ".venv" / "keep.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")

    result = uninstall.uninstall(
        candidate, dry_run=False, executable=tmp_path / "python"
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_skill_directory"
    assert marker.read_text(encoding="utf-8") == "keep"
