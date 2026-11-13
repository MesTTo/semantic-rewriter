"""Purpose: run every command README.md and examples/README.md show and write what each printed into them, between
`<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->`, so their transcripts are the code's output; a
block naming a knowledge base gets that file's text, and a block naming a function gets the markdown it returns. A
block may appear in both documents, and is rendered once.

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
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

# Progress bars from model loading would land in the transcripts.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from semrewrite.cli import main

# The documents whose generated blocks the walkthrough fills.
DOCUMENTS = (ROOT / "README.md", ROOT / "examples" / "README.md")
TOM = "examples/parent_of_tom.kb"
SWANSON = "examples/swanson.metta"
FAMILY = "examples/family_tree.metta"
VICTORIA = "examples/wikidata/victoria.metta"
WORTH = "worth_testing_against(X, raynauds)"
# The NLI cross-encoder judging every predicate of the small example, at its calibrated threshold.
NLI = ["--judge", "nli-deberta-large", "--retriever", "all"]
# The settings RESULTS.md found best for Jev: the threshold calibrated for answer F1, and a bound of three rewrites.
JEV = ["--judge", "jev", "--threshold", "0.65", "--strategy", "rewrite-depth-tabled", "--retriever", "all",
       "--engine", "pymetta"]
SIMILARITY = ["--judge", "cosine-bge", "--strategy", "rewrite-depth-tabled", "--retriever", "all", "--engine", "pymetta"]

# A command whose last word is this runs with its standard error discarded, as a shell would, and shows it typed.
QUIET = "2>/dev/null"



def wikidata(depth: int, judge: Sequence[str] = JEV[:4]) -> list[str]:
    """A judge's settings on Wikidata's catalogue, Jev's unless `judge` names another, as in
    examples/wikidata/evaluate.py: the cross-encoder ranks every property, the judge decides on the 50 it ranks highest,
    a path takes at most `depth` rewrites, and the work budget is the evaluation's, with no time limit."""
    return [*judge, "--strategy", "rewrite-depth-tabled", "--depth", str(depth), "--retriever", "judge", "--k", "50",
            "--engine", "pymetta", "--work", "50000000", "--seconds", "inf"]


@cache
def _evaluation():
    """examples/wikidata/evaluate.py as a module, loaded once, so the blocks it writes share its runs."""
    spec = importlib.util.spec_from_file_location("wikidata_evaluate", ROOT / "examples" / "wikidata" / "evaluate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wikidata_evaluation() -> str:
    """examples/wikidata/evaluate.py's report on every question."""
    return _evaluation().report([])


def wikidata_summary() -> str:
    """examples/wikidata/evaluate.py's table of every question, each judge allowed one rewrite on a path."""
    return _evaluation().summary([])


def wikidata_placed() -> str:
    """Where the answers outside Wikidata's land among Victoria's descendants ranked by their opinions, Jev admitting at
    rewrite threshold 0.1 (examples/wikidata/evaluate.py)."""
    return _evaluation().placed("victoria", 0.1)


# Each generated block: the paths of the knowledge bases it shows, space separated, the commands whose transcripts it
# holds, or a function writing it.
BLOCKS: dict[str, str | list[list[str]] | Callable[[], str]] = {
    "tom-source": TOM,
    "tom-explain": [["ask", TOM, "parent(X, tom)", *NLI, "--explain"]],
    "tom-strict": [["ask", TOM, "parent(X, tom)", *NLI, "--threshold", "0.995"]],
    "tom-grandparent": [["ask", TOM, "grandparent(X, tom)", *NLI, "--threshold", "0.995", "--explain"]],
    "wikidata-questions": "examples/wikidata/victoria.question.metta examples/wikidata/leonardo.question.metta "
                          "examples/wikidata/alphabet.question.metta",
    "wikidata-trace": [["ask", "examples/wikidata/alphabet.metta", "owns(alphabet_inc, youtube_q866)", *wikidata(1),
                        "--trace", "--explain"]],
    "wikidata-summary": wikidata_summary,
    "wikidata-evaluation": wikidata_evaluation,
    "wikidata-descent": [["ask", VICTORIA, "descendant_of(frederik_x_of_denmark, queen_victoria)", *wikidata(1),
                          "--explain"]],
    "wikidata-work": [["ask", "examples/wikidata/leonardo.metta", "created(leonardo_da_vinci, mona_lisa)", *wikidata(1),
                       "--explain"]],
    "wikidata-holding": [["ask", "examples/wikidata/alphabet.metta", "owns(alphabet_inc, youtube_q866)", *wikidata(1),
                          "--explain"]],
    "wikidata-similarity": [["ask", VICTORIA, "descendant_of(queen_victoria, queen_victoria)",
                             *wikidata(1, SIMILARITY[:2]), "--explain"]],
    "wikidata-placed": wikidata_placed,
    "wikidata-graded": [["ask", VICTORIA, "descendant_of(british_army, queen_victoria)",
                         *wikidata(1, ["--judge", "jev", "--threshold", "0.1"]), "--beliefs", "--explain", QUIET]],
    "wikidata-chained": [
        ["ask", VICTORIA, "descendant_of(army_air_corps, queen_victoria)", *wikidata(3), "--explain"],
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


def command_line(argv: list[str]) -> str:
    """The command as a shell user types it."""
    if argv[-1:] == [QUIET]:
        return f"semrewrite {shlex.join(argv[:-1])} {QUIET}"
    return f"semrewrite {shlex.join(argv)}"


def transcript(argv: list[str]) -> str:
    """The command line and everything it printed, standard output and standard error in the order written, as a
    terminal shows them; standard output alone for a command ending in 2>/dev/null."""
    quiet = argv[-1:] == [QUIET]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO() if quiet else printed):
        status = main(argv[:-1] if quiet else argv)
    if status == 1:
        msg = f"{command_line(argv)} failed: {printed.getvalue()}"
        raise SystemExit(msg)
    return f"$ {command_line(argv)}\n{printed.getvalue()}".rstrip("\n")


def render(name: str) -> str:
    """The block's new body: a function's markdown, the knowledge bases in a fence, MeTTa's when every one is a MeTTa
    file, or the commands' transcripts in a text fence."""
    spec = BLOCKS[name]
    if isinstance(spec, str):
        paths = spec.split()
        fence = "metta" if all(path.endswith(".metta") for path in paths) else "text"
        return f"```{fence}\n" + "\n\n".join((ROOT / path).read_text().rstrip() for path in paths) + "\n```"
    if isinstance(spec, list):
        return "```text\n" + "\n\n".join(transcript(argv) for argv in spec) + "\n```"
    return spec()


def _block(name: str) -> re.Pattern[str]:
    """A generated block's markers and whatever they hold."""
    return re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END GENERATED: "
                      rf"{re.escape(name)} -->)", re.DOTALL)


# Rendering can take an hour, so each document is read again only after every block is rendered, and an edit made to
# one meanwhile survives.
def main_walkthrough(names: list[str]) -> None:
    unknown = sorted(set(names) - set(BLOCKS))
    if unknown:
        msg = f"no generated block is named {', '.join(unknown)}; the blocks are {', '.join(BLOCKS)}"
        raise SystemExit(msg)
    chosen = list(names or BLOCKS)
    before = [document.read_text() for document in DOCUMENTS]
    missing = [name for name in chosen if not any(_block(name).search(text) for text in before)]
    if missing:
        msg = f"no document has a generated block named {', '.join(missing)}"
        raise SystemExit(msg)
    bodies = {}
    for name in chosen:
        started = time.perf_counter()
        bodies[name] = render(name)
        print(f"rendered {name} in {time.perf_counter() - started:.1f} s", file=sys.stderr)
    for document in DOCUMENTS:
        text = document.read_text()
        for name, body in bodies.items():
            text = _block(name).sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
        document.write_text(text)


if __name__ == "__main__":
    main_walkthrough(sys.argv[1:])
