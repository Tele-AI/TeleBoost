"""Repository-wide dependency direction gates.

These checks deliberately inspect source without importing it.  That keeps the
architecture contract dependency-light and catches imports hidden inside
functions, optional branches, and concrete backend modules.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent


@dataclass(frozen=True, order=True)
class _Violation:
    path: str
    line: int
    target: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.target}"


@dataclass(frozen=True)
class _Family:
    name: str
    roots: tuple[Path, ...]
    import_prefixes: tuple[str, ...]


_FAMILIES = (
    _Family(
        name="wan",
        roots=(
            REPO_ROOT / "teleboost" / "models" / "wan",
            REPO_ROOT / "teleboost" / "programs" / "wan",
            REPO_ROOT / "teleboost" / "training" / "families" / "wan",
        ),
        import_prefixes=(
            "teleboost.models.wan.teletron",
            "teleboost.models.wan.dual",
            "teleboost.models.wan.family",
            "teleboost.programs.wan",
            "teleboost.training.families.wan",
            "wan",
        ),
    ),
)

_CONCRETE_FAMILY_PREFIXES = tuple(prefix for family in _FAMILIES for prefix in family.import_prefixes)


def _python_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    yield from sorted(path for path in root.rglob("*.py") if path.is_file())


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from_import(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    module = _module_name(path)
    package_parts = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    keep = len(package_parts) - (node.level - 1)
    if keep < 0:
        return node.module or ""
    target_parts = package_parts[:keep]
    if node.module:
        target_parts.extend(node.module.split("."))
    return ".".join(target_parts)


def _imports(path: Path) -> Iterator[tuple[int, str]]:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from_import(path, node)
            for alias in node.names:
                if not module:
                    continue
                target = module if alias.name == "*" else f"{module}.{alias.name}"
                yield node.lineno, target


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _violations_for_import_prefixes(
    roots: Iterable[Path],
    forbidden_prefixes: Iterable[str],
) -> list[_Violation]:
    prefixes = tuple(forbidden_prefixes)
    violations = {
        _Violation(
            path=path.relative_to(REPO_ROOT).as_posix(),
            line=line,
            target=target,
        )
        for root in roots
        for path in _python_files(root)
        for line, target in _imports(path)
        if any(_matches_prefix(target, prefix) for prefix in prefixes)
    }
    return sorted(violations)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            docstrings.add(id(first.value))
    return docstrings


def _dynamic_recipe_references(root: Path) -> list[_Violation]:
    violations: set[_Violation] = set()
    for path in _python_files(root):
        tree = _tree(path)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if id(node) in docstrings:
                continue
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            value = node.value.strip()
            if value == "recipes" or value.startswith(("recipes.", "recipes/")):
                violations.add(
                    _Violation(
                        path=path.relative_to(REPO_ROOT).as_posix(),
                        line=node.lineno,
                        target=repr(value),
                    )
                )
    return sorted(violations)


def _assert_clean(rule: str, violations: list[_Violation]) -> None:
    details = "\n".join(f"  - {violation.render()}" for violation in violations)
    assert not violations, f"{rule}\n{details}"


def test_teleboost_does_not_import_recipes() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "teleboost",),
        ("recipe", "recipes"),
    )

    _assert_clean("teleboost must not import the recipes composition layer", violations)


def test_teleboost_has_no_dynamic_recipe_module_references() -> None:
    violations = _dynamic_recipe_references(REPO_ROOT / "teleboost")

    _assert_clean(
        "teleboost must not hide recipes dependencies in dynamic module/default strings",
        violations,
    )


def test_removed_production_roots_are_not_reintroduced() -> None:
    removed_roots = (
        REPO_ROOT / "teleboost" / "backends",
        REPO_ROOT / "teleboost" / "integrations",
        REPO_ROOT / "teleboost" / "runtime",
        REPO_ROOT / "teleboost" / "tools",
        REPO_ROOT / "teleboost" / "train",
        REPO_ROOT / "teleboost" / "workers",
        REPO_ROOT / "teleboost" / "vllm_bootstrap",
        REPO_ROOT / "teleboost" / "datasets" / "transform",
        REPO_ROOT / "teleboost" / "models" / "context_parallel",
        REPO_ROOT / "teleboost" / "models" / "wan_teletron",
        REPO_ROOT / "teleboost" / "reward" / "functions",
        REPO_ROOT / "teleboost" / "reward" / "reward_models",
        REPO_ROOT / "data_preprocess",
        REPO_ROOT / "recipe",
    )
    removed_files = (
        REPO_ROOT / "teleboost" / "programs" / "wan" / "diag_helpers.py",
        REPO_ROOT / "teleboost" / "programs" / "wan" / "diffusion_rollout.py",
        REPO_ROOT / "teleboost" / "programs" / "wan" / "dp_actor.py",
        REPO_ROOT / "teleboost" / "programs" / "wan" / "fsdp_worker.py",
        REPO_ROOT / "teleboost" / "programs" / "wan" / "runtime_bootstrap.py",
    )
    present = [path.relative_to(REPO_ROOT).as_posix() for path in (*removed_roots, *removed_files) if path.exists()]

    assert not present, f"removed production roots were reintroduced: {present}"


def _visible_directories(relative: str) -> set[str]:
    return {child.name for child in (REPO_ROOT / relative).iterdir() if child.is_dir() and child.name != "__pycache__"}


def test_public_source_tree_contains_only_the_wan_family() -> None:
    assert _visible_directories("teleboost/models") == {"wan"}
    assert _visible_directories("teleboost/programs") == {"wan"}
    assert _visible_directories("teleboost/training/families") == {"wan"}
    assert _visible_directories("teleboost/config/presets/programs") == {"wan_grpo_fsdp"}
    assert _visible_directories("third_party") == {"wan", "raft", "Videophy"}

    from teleboost.programs.builtins import BUILTIN_PROGRAMS

    assert BUILTIN_PROGRAMS
    assert {program.family for program in BUILTIN_PROGRAMS} == {"wan"}
    assert {program.backend_name for program in BUILTIN_PROGRAMS} == {"wan"}


def test_public_program_and_algorithm_surfaces_match_the_release_scope() -> None:
    from teleboost.programs.builtins import BUILTIN_PROGRAMS

    algorithms = {program.algorithm for program in BUILTIN_PROGRAMS}
    assert algorithms == {"grpo", "bgpo", "vipo", "tempflow", "dpo"}
    assert _visible_directories("recipes") == {
        "wan_grpo_fsdp",
        "wan_bgpo_fsdp",
        "wan_vipo_fsdp",
        "wan_tempflow_fsdp",
        "wan_dpo_teletron",
    }

    algorithm_entries = {entry.name for entry in (REPO_ROOT / "teleboost/algorithms").iterdir() if entry.name != "__pycache__"}
    assert algorithm_entries == {
        "README.md",
        "__init__.py",
        "bgpo.py",
        "grpo",
        "grpo_guard.py",
        "rollout_contract.py",
        "solver_contract.py",
        "tempflow",
        "vipo.py",
        "wan_transition.py",
    }


def test_training_core_does_not_import_concrete_families() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "teleboost" / "training" / "core",),
        _CONCRETE_FAMILY_PREFIXES,
    )

    _assert_clean("teleboost.training.core must remain model-family neutral", violations)


def test_training_core_does_not_import_family_training_adapters() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "teleboost" / "training" / "core",),
        ("teleboost.training.families",),
    )

    _assert_clean("teleboost.training.core must receive family hooks by injection", violations)


def test_training_families_do_not_import_program_composition_root() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "teleboost" / "training" / "families",),
        ("teleboost.programs",),
    )

    _assert_clean("teleboost.training.families must not depend on program composition", violations)


def test_teletron_framework_does_not_import_models_or_training() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "teleboost" / "engines" / "teletron",),
        (
            "teleboost.models",
            "teleboost.training",
        ),
    )

    _assert_clean(
        "teleboost.engines.teletron must receive concrete model/training hooks by injection",
        violations,
    )


def test_fsdp_engine_is_the_only_production_fsdp_dispatch_import() -> None:
    violations = []
    legacy_prefixes = (
        "teleboost.training.core.fsdp",
        "teleboost.integrations.verl_fsdp_merge",
    )
    for root in (REPO_ROOT / "teleboost", REPO_ROOT / "recipes"):
        for path in _python_files(root):
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            for line, target in _imports(path):
                if any(_matches_prefix(target, prefix) for prefix in legacy_prefixes):
                    violations.append(_Violation(rel_path, line, target))

    _assert_clean(
        "production FSDP dispatch and checkpoint merge imports must use teleboost.engines.fsdp",
        sorted(violations),
    )


def test_transport_and_sharding_helpers_import_from_engines() -> None:
    violations = []
    legacy_prefixes = (
        "teleboost.runtime.transfer_queue",
        "teleboost.workers.sharding_manager",
    )
    for root in (REPO_ROOT / "teleboost", REPO_ROOT / "recipes"):
        for path in _python_files(root):
            rel_path = path.relative_to(REPO_ROOT).as_posix()
            for line, target in _imports(path):
                if any(_matches_prefix(target, prefix) for prefix in legacy_prefixes):
                    violations.append(_Violation(rel_path, line, target))

    _assert_clean(
        "production transport and sharding imports must use teleboost.engines",
        sorted(violations),
    )


def test_reward_layer_does_not_import_program_composition_root() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "teleboost" / "reward",),
        ("teleboost.programs",),
    )

    _assert_clean(
        "teleboost.reward must not depend on the program composition root",
        violations,
    )


def test_concrete_families_do_not_import_each_other() -> None:
    violations: list[_Violation] = []
    for source_family in _FAMILIES:
        foreign_prefixes = tuple(prefix for target_family in _FAMILIES if target_family.name != source_family.name for prefix in target_family.import_prefixes)
        violations.extend(_violations_for_import_prefixes(source_family.roots, foreign_prefixes))

    _assert_clean("concrete model families must not import one another", sorted(set(violations)))


def test_third_party_does_not_import_teleboost() -> None:
    violations = _violations_for_import_prefixes(
        (REPO_ROOT / "third_party",),
        ("teleboost",),
    )

    _assert_clean("vendored third_party code must not import its TeleBoost host", violations)
