import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
    return roots


def test_shared_context_parallel_has_no_model_specific_imports():
    shared_dirs = (
        ROOT / "teleboost/engines/fsdp/context_parallel",
        ROOT / "teleboost/engines/teletron/context_parallel",
    )
    forbidden = (
        "teleboost.models.wan",
        "teleboost.models.wan.teletron",
        "teleboost.models.wan.family",
        "recipes.",
    )

    offenders = []
    for shared_dir in shared_dirs:
        for path in shared_dir.rglob("*.py"):
            for module in _import_roots(path):
                if module.startswith(forbidden):
                    offenders.append(f"{path.relative_to(ROOT)}:{module}")

    assert offenders == []
