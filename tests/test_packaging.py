"""Purpose: tests that the core package needs none of its optional libraries.
With every library an extra installs blocked, each module but the PyMeTTa
engine imports, the command line answers the example queries with the
pure-Python engine and the oracle judge, and a command needing a missing
library names the extra that installs it. The command line's table of extras
matches pyproject.toml.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

from semrewrite.cli import EXTRAS

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLE = "examples/parent_of_tom.kb"

# A None entry in sys.modules makes `import name` raise ModuleNotFoundError, as if the library were absent.
BLOCK = f"import sys\nfor name in {tuple(EXTRAS)!r}:\n    sys.modules[name] = None\n"


def blocked(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter at the repository root with every optional library blocked."""
    return subprocess.run([sys.executable, "-c", BLOCK + code], cwd=ROOT, capture_output=True, text=True, check=False)


def cli(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return blocked(f"from semrewrite.cli import main\nraise SystemExit(main({argv!r}))")


def test_every_module_but_the_pymetta_engine_imports_without_optional_libraries() -> None:
    modules = sorted(
        ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
        for path in (ROOT / "semrewrite").rglob("*.py")
    )
    code = (
        "import importlib\n"
        f"for m in {modules!r}:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except ModuleNotFoundError as e:\n"
        "        print(m, e.name)\n"
    )
    run = blocked(code)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == ["semrewrite.engines.pymetta metta"]
    assert len(modules) > 20


@pytest.mark.parametrize(
    ("query", "answers"),
    [
        ("parent(X, tom)", ["parent(alice, tom)", "parent(bob, tom)"]),
        ("grandparent(X, tom)", ["grandparent(carol, tom)", "grandparent(dave, tom)"]),
    ],
)
def test_command_line_answers_without_optional_libraries(query: str, answers: list[str]) -> None:
    run = cli(["ask", EXAMPLE, query, "--judge", "oracle", "--retriever", "all", "--engine", "python"])
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines() == answers


@pytest.mark.parametrize(
    ("argv", "missing", "extra"),
    [
        (["conformance", "--engine", "python", "--examples", "5"], "hypothesis", "conformance"),
        (["ask", EXAMPLE, "parent(X, tom)", "--judge", "nli-deberta-large", "--retriever", "all"],
         "sentence_transformers", "models"),
        (["ask", EXAMPLE, "parent(X, tom)", "--judge", "oracle"], "sentence_transformers", "models"),
        (["ask", EXAMPLE, "parent(X, tom)", "--judge", "oracle", "--retriever", "all", "--engine", "pymetta"],
         "metta", "pymetta"),
    ],
)
def test_a_missing_library_names_the_extra_that_installs_it(argv: list[str], missing: str, extra: str) -> None:
    run = cli(argv)
    assert run.returncode == 1
    assert run.stderr.strip() == f"semrewrite: {missing} is not installed; pip install 'semantic-rewriter[{extra}]'"


def test_command_line_honours_the_oracle_threshold_and_names_a_strategy_the_engine_cannot_run() -> None:
    base = ["ask", EXAMPLE, "parent(X, tom)", "--judge", "oracle", "--retriever", "all", "--engine", "python"]
    run = cli([*base, "--threshold", "1.5"])
    assert run.returncode == 0 and run.stdout == "", run.stdout
    run = cli([*base, "--strategy", "naive"])
    assert run.returncode == 1 and "Traceback" not in run.stderr
    assert run.stderr.strip() == "semrewrite: engine python cannot run naive: it lacks untabled"


def test_a_judge_closes_without_torch() -> None:
    """Jev's extra installs no torch, so closing a judge, by hand or by leaving its with block, must not import it."""
    run = blocked("from semrewrite.judges.jev import JevJudge\n"
                  "JevJudge('jev', 0.5).close()\n"
                  "with JevJudge('jev', 0.5):\n"
                  "    pass\n"
                  "print('closed')\n")
    assert run.returncode == 0 and run.stdout.strip() == "closed", run.stderr


def test_extras_table_matches_pyproject() -> None:
    """Each installed library an extra lists is in the table under that extra, and each table entry is so listed."""
    def canonical(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    listed = {
        extra: {canonical(re.match(r"[A-Za-z0-9._-]+", r)[0]) for r in requirements}
        for extra, requirements in extras.items()
        if extra != "all"
    }
    assert set(EXTRAS.values()) == set(listed)
    provided: dict[str, set[str]] = {}
    for module, distributions in importlib.metadata.packages_distributions().items():
        for distribution in distributions:
            provided.setdefault(canonical(distribution), set()).add(module)
    checked = 0
    for extra, distributions in listed.items():
        for distribution in distributions & provided.keys():
            assert any(EXTRAS.get(m) == extra for m in provided[distribution]), (extra, distribution)
            checked += 1
    for module, extra in EXTRAS.items():
        owners = {d for d, modules in provided.items() if module in modules}
        assert not owners or owners & listed[extra], (module, extra, owners)
    if not checked:
        pytest.skip("no optional library is installed here")
