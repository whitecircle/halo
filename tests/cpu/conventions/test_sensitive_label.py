"""The sensitive-label check fails a pull request that changes a listed path without the label, judged by the
list on the branch it merges into, and every glob in ``.github/sensitive-paths`` still names a file."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.common.repo import REPO_ROOT, load_script_module

_SCRIPT = ".github/scripts/check_sensitive_label.py"
label_check = load_script_module(_SCRIPT)
_BASE_BRANCH = "main"


def _repo_files() -> list[str]:
    files = []
    for directory, subdirectories, names in os.walk(REPO_ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in (".git", "__pycache__")]
        files += [str((Path(directory) / name).relative_to(REPO_ROOT)) for name in names]
    return files


@pytest.mark.parametrize(
    ("glob", "path", "matches"),
    [
        ("CLAUDE.md", "CLAUDE.md", True),
        ("CLAUDE.md", "notes/CLAUDE.md", False),
        ("Dockerfile*", "Dockerfile", True),
        ("Dockerfile*", "Dockerfile.vllm", True),
        ("Dockerfile*", "build/Dockerfile.base", False),
        ("docker-compose*.yml", "docker-compose.sglang.efa.yml", True),
        ("docker-compose*.yml", "docker-compose.yml.bak", False),
        ("build/**", "build/images/patches/patch.py", True),
        ("build/**", "build.md", False),
        ("core/**", "core_extra/module.py", False),
        ("core/*.py", "core/module.py", True),
        ("core/*.py", "core/nested/module.py", False),
        ("**/module.py", "module.py", True),
        ("**/module.py", "core/nested/module.py", True),
        ("core/**/module.py", "core/module.py", True),
        ("core/**/module.py", "core/a/b/module.py", True),
        ("core/**/module.py", "other/core/module.py", False),
        ("a.b", "axb", False),
    ],
)
def test_star_stays_in_a_segment_and_double_star_spans_any(glob, path, matches):
    assert (label_check.glob_regex(glob).fullmatch(path) is not None) is matches


def test_the_list_covers_the_rulebook_and_every_glob_names_a_file():
    """A glob that names nothing protects nothing: a listed path that moved must be re-keyed in the move."""
    globs = label_check.globs_in((REPO_ROOT / label_check.SENSITIVE_PATHS).read_text(encoding="utf-8"))
    assert {"CLAUDE.md", "AGENTS.md", label_check.SENSITIVE_PATHS} <= set(globs), globs
    files = _repo_files()
    assert len(files) > 1000, f"the walk found only {len(files)} files"
    stale = [glob for glob in globs if not label_check.sensitive_changes(files, [glob])]
    assert not stale, f"{label_check.SENSITIVE_PATHS} lists globs no file matches: {stale}"


def _git(repo: Path, *args: str) -> str:
    identity = ("-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false")
    return subprocess.run(
        ["git", *identity, *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str] | None = None) -> str:
    for relative, text in (files or {}).items():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path) -> Path:
    """A repository whose list protects ``core/``, its rulebook and itself, with the check where the workflow
    runs it, and a pull request branch forked from its base branch."""
    _git(tmp_path, "init", "-q", "-b", _BASE_BRANCH)
    _commit(
        tmp_path,
        {
            _SCRIPT: (REPO_ROOT / _SCRIPT).read_text(encoding="utf-8"),
            label_check.SENSITIVE_PATHS: f"# protected\ncore/**\nRULES.md  # the rulebook\n{label_check.SENSITIVE_PATHS}\n",
            "core/engine.py": "ENGINE = 1\n",
            "app.py": "APP = 1\n",
            "RULES.md": "# Rules\n",
        },
    )
    _git(tmp_path, "checkout", "-q", "-b", "pull-request")
    return tmp_path


def _check(repo: Path, head: str, labels: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / _SCRIPT), _BASE_BRANCH, head, "--labels", json.dumps(labels)],
        capture_output=True,
        text=True,
        check=False,
    )


def _refused(result: subprocess.CompletedProcess) -> list[str]:
    """The paths a refusal lists; fails unless the check refused."""
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stderr.startswith("::error::"), result.stderr
    return [line.strip() for line in result.stderr.splitlines()[1:]]


def test_a_change_outside_the_list_passes(repo):
    result = _check(repo, _commit(repo, {"app.py": "APP = 2\n"}), [])
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_a_change_to_a_listed_path_needs_the_label(repo):
    head = _commit(repo, {"core/engine.py": "ENGINE = 2\n", "RULES.md": "# Rules, amended\n"})
    assert _refused(_check(repo, head, ["bug"])) == ["RULES.md", "core/engine.py"]
    labeled = _check(repo, head, ["bug", label_check.LABEL])
    assert labeled.returncode == 0, labeled.stderr


def test_a_file_moved_out_of_a_listed_directory_counts_under_its_old_path(repo):
    _git(repo, "mv", "core/engine.py", "engine.py")
    assert _refused(_check(repo, _commit(repo), [])) == ["core/engine.py"]


def test_a_pull_request_is_judged_by_the_base_branch_list(repo):
    """Dropping a protected directory from the list in the same pull request that edits it still needs the label."""
    head = _commit(repo, {label_check.SENSITIVE_PATHS: "RULES.md\n", "core/engine.py": "ENGINE = 2\n"})
    assert _refused(_check(repo, head, [])) == [label_check.SENSITIVE_PATHS, "core/engine.py"]


def test_only_the_pull_requests_own_changes_count(repo):
    """A listed path the base branch changed after the fork is not the pull request's change."""
    head = _commit(repo, {"app.py": "APP = 2\n"})
    _git(repo, "checkout", "-q", _BASE_BRANCH)
    _commit(repo, {"core/engine.py": "ENGINE = 3\n"})
    result = _check(repo, head, [])
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
