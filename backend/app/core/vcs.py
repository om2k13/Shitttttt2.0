import tempfile, shutil, os, subprocess, pathlib
from pathlib import Path
from git import Repo
from .settings import settings

def prepare_workspace(job_id: str) -> Path:
    base = Path(settings.WORK_DIR)
    base.mkdir(parents=True, exist_ok=True)
    path = base / job_id
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def clone_repo(job_id: str, repo_url: str, branch: str | None = None) -> Path:
    path = prepare_workspace(job_id)
    Repo.clone_from(repo_url, path)
    if branch:
        repo = Repo(path)
        repo.git.checkout(branch)
    return path

def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err
