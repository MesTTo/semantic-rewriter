"""Purpose: run every command examples/README.md shows and write what each printed into the README, between
`<!-- BEGIN GENERATED: name -->` and `<!-- END GENERATED: name -->`, so its transcripts are the code's output; a
block naming a knowledge base gets that file's text.

Run from the repository root:

    python examples/walkthrough.py [BLOCK ...]

Naming blocks reruns only those.

The commands ask Jev, so a question its answer cache lacks needs a TypeSafe key in TYPESAFE_API_KEY, or the path of a
file holding one in JEV_KEY_FILE; the cache stays in the repository it was made in. The similarity judge's commands
need the models extra, and every command needs the pymetta extra.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import sys
import time
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

# Each generated block: a knowledge base's path, or the commands whose transcripts it holds.
BLOCKS: dict[str, str | list[list[str]]] = {
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
    """The block's new body: the knowledge base in a MeTTa fence, or the commands' transcripts in a text fence."""
    spec = BLOCKS[name]
    if isinstance(spec, str):
        return f"```metta\n{(ROOT / spec).read_text().rstrip()}\n```"
    return "```text\n" + "\n\n".join(transcript(argv) for argv in spec) + "\n```"


def main_walkthrough(names: list[str]) -> None:
    unknown = sorted(set(names) - set(BLOCKS))
    if unknown:
        msg = f"no generated block is named {', '.join(unknown)}; the blocks are {', '.join(BLOCKS)}"
        raise SystemExit(msg)
    text = README.read_text()
    for name in names or BLOCKS:
        pattern = re.compile(rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END GENERATED: "
                             rf"{re.escape(name)} -->)", re.DOTALL)
        if not pattern.search(text):
            msg = f"{README} has no generated block named {name}"
            raise SystemExit(msg)
        started = time.perf_counter()
        body = render(name)
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
        print(f"rendered {name} in {time.perf_counter() - started:.1f} s", file=sys.stderr)
    README.write_text(text)


if __name__ == "__main__":
    main_walkthrough(sys.argv[1:])
