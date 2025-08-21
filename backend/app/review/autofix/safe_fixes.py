from pathlib import Path
from ...core.vcs import run_cmd
from typing import List

def run_python_formatters(repo: Path) -> list[tuple[str,int,str]]:
    results = []
    rc, out, err = run_cmd(["ruff","check","--fix","."], repo)
    results.append(("ruff", rc, out or err))
    rc, out, err = run_cmd(["black","-q","."], repo)
    results.append(("black", rc, out or err))
    return results

def run_js_formatters(repo: Path) -> list[tuple[str,int,str]]:
    results = []
    if (repo / "package.json").exists():
        rc, out, err = run_cmd(["npx","eslint","--fix","."], repo)
        results.append(("eslint", rc, out or err))
        rc, out, err = run_cmd(["npx","prettier","-w","."], repo)
        results.append(("prettier", rc, out or err))
    return results
