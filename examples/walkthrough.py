"""Purpose: run every command examples/README.md shows and write what each printed into the README, between
`<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->`, so its transcripts are the code's output; a
block naming a knowledge base gets that file's text, and a block naming a function gets the markdown it returns.

Run from the repository root:

    python examples/walkthrough.py [BLOCK ...]

Naming blocks reruns only those.

The commands ask Jev, so a question its answer cache lacks needs a TypeSafe key in TYPESAFE_API_KEY, or the path of a
file holding one in JEV_KEY_FILE; the cache stays in the repository it was made in. The similarity judge's commands
need the models extra, and every command needs the pymetta extra.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Progress bars from model loading would land in the transcripts.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from semrewrite.cli import main

README = ROOT / "examples" / "README.md"
SWANSON = "examples/swanson.metta"
FAMILY = "examples/family_tree.metta"
WORTH = "worth_testing_against(X, raynauds)"
# The settings RESULTS.md found best for Jev: the threshold calibrated for answer F1, and a bound of three rewrites.
JEV = ["--judge", "jev", "--threshold", "0.65", "--strategy", "rewrite-depth-tabled", "--retriever", "all",
       "--engine", "pymetta"]
SIMILARITY = ["--judge", "cosine-bge", "--strategy", "rewrite-depth-tabled", "--retriever", "all", "--engine", "pymetta"]


def wikidata(depth: int) -> list[str]:
    """Jev's settings on Wikidata's catalogue, as in examples/wikidata/evaluate.py: the cross-encoder ranks every
    property, Jev judges the 50 it ranks highest, a path takes at most `depth` rewrites, and the work budget is the
    evaluation's, with no time limit."""
    return [*JEV[:6], "--depth", str(depth), "--retriever", "judge", "--k", "50", "--engine", "pymetta",
            "--work", "50000000", "--seconds", "inf"]


def wikidata_evaluation() -> str:
    """examples/wikidata/evaluate.py's report on every question."""
    spec = importlib.util.spec_from_file_location("wikidata_evaluate", ROOT / "examples" / "wikidata" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.report([])


# Each generated block: the paths of the knowledge bases it shows, space separated, the commands whose transcripts it
# holds, or a function writing it.
BLOCKS: dict[str, str | list[list[str]] | Callable[[], str]] = {
    "wikidata-questions": "examples/wikidata/victoria.question.metta examples/wikidata/leonardo.question.metta "
                          "examples/wikidata/alphabet.question.metta",
    "wikidata-evaluation": wikidata_evaluation,
    "wikidata-derivations": [
        ["ask", "examples/wikidata/victoria.metta", "descendant_of(frederik_x_of_denmark, queen_victoria)", *wikidata(1),
         "--explain"],
        ["ask", "examples/wikidata/leonardo.metta", "created(leonardo_da_vinci, mona_lisa)", *wikidata(1), "--explain"],
        ["ask", "examples/wikidata/alphabet.metta", "owns(alphabet_inc, youtube_q866)", *wikidata(1), "--explain"],
    ],
    "wikidata-chained": [
        ["ask", "examples/wikidata/victoria.metta", "descendant_of(army_air_corps, queen_victoria)", *wikidata(3),
         "--explain"],
    ],
    "swanson-source": SWANSON,
    "swanson-exact": [["ask", SWANSON, WORTH, "--strategy", "exact", "--engine", "pymetta"]],
    "swanson-jev": [["ask", SWANSON, WORTH, *JEV, "--explain"]],
    "swanson-paths": [["ask", SWANSON, q, *JEV] for q in ("lowers(omega_3, B)", "aggravates(B, raynauds)")],
    "swanson-judged": [["rewrites", SWANSON, p, "--judge", "jev", "--threshold", "0.65", "--retriever", "all", "--all"]
                       for p in ("lowers", "aggravates")],
    "swanson-similarity": [["ask", SWANSON, WORTH, *SIMILARITY, "--explain"]],
    # A time budget the work budget always reaches first, so the transcript shows the same stop on any machine.
    "swanson-naive": [["ask", SWANSON, WORTH, *JEV[:4], "--strategy", "naive", "--retriever", "all",
                       "--engine", "pymetta", "--seconds", "600"]],
    "family-source": FAMILY,
    "family-jev": [["ask", FAMILY, "descendant_of(X, tom)", *JEV, "--explain"]],
    "family-ancestors": [["ask", FAMILY, "descendant_of(jim, Y)", *JEV]],
    "family-mothers": [["ask", FAMILY, q, *JEV, "--explain"] for q in ("mother_of(X, bob)", "mother_of(X, jim)")],
    "family-judged": [["rewrites", FAMILY, "mother_of", "--judge", "jev", "--threshold", "0.65", "--retriever", "all",
                       "--all"]],
    "family-similarity": [["ask", FAMILY, q, *SIMILARITY, "--explain"]
                          for q in ("descendant_of(X, tom)", "mother_of(X, jim)")],
}


def transcript(argv: list[str]) -> str:
    """The command line and everything it printed, its answers first, then what it wrote to stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = main(argv)
    if status == 1:
        msg = f"semrewrite {shlex.join(argv)} failed: {err.getvalue()}"
        raise SystemExit(msg)
    return f"$ semrewrite {shlex.join(argv)}\n{out.getvalue()}{err.getvalue()}".rstrip("\n")


def render(name: str) -> str:
    """The block's new body: a function's markdown, the knowledge bases in a MeTTa fence, or the commands'
    transcripts in a text fence."""
    spec = BLOCKS[name]
    if isinstance(spec, str):
        return "```metta\n" + "\n\n".join((ROOT / path).read_text().rstrip() for path in spec.split()) + "\n```"
    if isinstance(spec, list):
        return "```text\n" + "\n\n".join(transcript(argv) for argv in spec) + "\n```"
    return spec()


def _block(name: str) -> re.Pattern[str]:
    """A generated block's markers and whatever they hold."""
    return re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END GENERATED: "
                      rf"{re.escape(name)} -->)", re.DOTALL)


# Rendering can take an hour, so the README is read again only after every block is rendered, and an edit made to
# it meanwhile survives.
def main_walkthrough(names: list[str]) -> None:
    unknown = sorted(set(names) - set(BLOCKS))
    if unknown:
        msg = f"no generated block is named {', '.join(unknown)}; the blocks are {', '.join(BLOCKS)}"
        raise SystemExit(msg)
    chosen = list(names or BLOCKS)
    before = README.read_text()
    missing = [name for name in chosen if not _block(name).search(before)]
    if missing:
        msg = f"{README} has no generated block named {', '.join(missing)}"
        raise SystemExit(msg)
    bodies = {}
    for name in chosen:
        started = time.perf_counter()
        bodies[name] = render(name)
        print(f"rendered {name} in {time.perf_counter() - started:.1f} s", file=sys.stderr)
    text = README.read_text()
    for name, body in bodies.items():
        text = _block(name).sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    README.write_text(text)


if __name__ == "__main__":
    main_walkthrough(sys.argv[1:])
