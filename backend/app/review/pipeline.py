import asyncio, json, os, shutil
from pathlib import Path
from typing import List, Dict
from sqlmodel import select
from ..core.vcs import clone_repo, run_cmd, cleanup_repo_after_review
from ..core.llm import enrich_findings_with_llm
from ..db.models import Job, Finding
from ..db.base import get_session

def _severity_map(level: str) -> str:
    s = (level or "").lower()
    if s in ("critical","high","medium","low"):
        return s
    if s in ("error","warn","warning"):
        return "medium"
    return "low"

async def run_review(job_id: str):
    print(f"🚀 Starting review for job: {job_id}")
    
    # Clean up old workspaces first
    from ..core.vcs import cleanup_old_workspaces
    cleanup_old_workspaces(max_age_hours=1)  # Clean up repos older than 1 hour
    
    # clone
    from ..db.base import SessionLocal
    async with SessionLocal() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if not job: 
            print(f"❌ Job {job_id} not found")
            return
        job.status, job.current_stage, job.progress = "running", "clone", 5
        await session.commit()
        print(f"📋 Job status updated to running")

    async with SessionLocal() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        print(f"🔗 Cloning repo: {job.repo_url} (branch: {job.base_branch})")
        repo_path = clone_repo(job_id, job.repo_url, job.base_branch)
        print(f"📁 Repo cloned to: {repo_path}")
        
        # Update progress
        job.current_stage, job.progress = "analyzing", 20
        await session.commit()

    findings: List[Dict] = []

    async def add(tool: str, items: List[Dict]):
        for it in items:
            findings.append({
                "job_id": job_id,
                "tool": tool,
                "severity": _severity_map(it.get("severity","low")),
                "file": it.get("file"),
                "line": it.get("line"),
                "rule_id": it.get("rule_id"),
                "message": it.get("message"),
                "remediation": it.get("remediation"),
                "autofixable": it.get("autofixable", False),
            })

    # --- Bandit (Python) ---
    print(f"🔍 Running Bandit security analysis...")
    rc, out, err = run_cmd(["bandit","-q","-r",".","-f","json"], repo_path)
    print(f"   Bandit exit code: {rc}, output length: {len(out)}, errors: {len(err)}")
    
    # Bandit exit codes: 0 = no issues, 1 = issues found, 2 = error
    if rc == 0:
        if out.strip():
            try:
                data = json.loads(out)
                findings_count = len(data.get("results", []))
                print(f"   Found {findings_count} Bandit issues")
                for issue in data.get("results", []):
                    await add("bandit", [{
                        "severity": _severity_map(issue.get("issue_severity","LOW")),
                        "file": issue.get("filename"),
                        "line": issue.get("line_number"),
                        "rule_id": issue.get("test_id"),
                        "message": issue.get("issue_text"),
                    }])
            except Exception as e:
                print(f"   Error parsing Bandit output: {e}")
        else:
            print("   ✅ Bandit found no security issues")
    elif rc == 1 and out.strip():
        # Exit code 1 means issues were found (this is good!)
        try:
            data = json.loads(out)
            findings_count = len(data.get("results", []))
            print(f"   ✅ Found {findings_count} Bandit security issues")
            for issue in data.get("results", []):
                await add("bandit", [{
                    "severity": _severity_map(issue.get("issue_severity","LOW")),
                    "file": issue.get("filename"),
                    "line": issue.get("line_number"),
                    "rule_id": issue.get("test_id"),
                    "message": issue.get("issue_text"),
                }])
        except Exception as e:
            print(f"   Error parsing Bandit output: {e}")
    elif rc == 1 and not out.strip():
        print("   ⚠️ Bandit found issues but produced no output")
    elif rc == 2:
        print(f"   ❌ Bandit failed to run: {err}")
    else:
        print(f"   ⚠️ Unexpected Bandit result: rc={rc}, output_length={len(out)}")

    # --- Semgrep ---
    print(f"🔍 Running Semgrep security analysis...")
    rc, out, err = run_cmd(["semgrep","--quiet","--json","--error","--timeout","0","--config","p/ci"], repo_path)
    print(f"   Semgrep exit code: {rc}, output length: {len(out)}, errors: {len(err)}")
    
    # Semgrep exit codes: 0 = no issues, 1 = issues found, 2 = error
    if rc == 0:
        if out.strip():
            try:
                data = json.loads(out)
                findings_count = len(data.get("results", []))
                print(f"   Found {findings_count} Semgrep issues")
                for r in data.get("results", []):
                    await add("semgrep", [{
                        "severity": r.get("extra",{}).get("severity","LOW"),
                        "file": r.get("path"),
                        "line": r.get("start",{}).get("line"),
                        "rule_id": r.get("check_id"),
                        "message": r.get("extra",{}).get("message"),
                    }])
            except Exception as e:
                print(f"   Error parsing Semgrep output: {e}")
        else:
            print("   ✅ Semgrep found no security issues")
    elif rc == 1 and out.strip():
        # Exit code 1 means issues were found (this is good!)
        try:
            data = json.loads(out)
            findings_count = len(data.get("results", []))
            print(f"   ✅ Found {findings_count} Semgrep security issues")
            for r in data.get("results", []):
                await add("semgrep", [{
                    "severity": r.get("extra",{}).get("severity","LOW"),
                    "file": r.get("path"),
                    "line": r.get("start",{}).get("line"),
                        "rule_id": r.get("check_id"),
                        "message": r.get("extra",{}).get("message"),
                }])
        except Exception as e:
            print(f"   Error parsing Semgrep output: {e}")
    elif rc == 1 and not out.strip():
        print("   ⚠️ Semgrep found issues but produced no output")
    elif rc == 2:
        print(f"   ❌ Semgrep failed to run: {err}")
    else:
        print(f"   ⚠️ Unexpected Semgrep result: rc={rc}, output_length={len(out)}")

    # --- detect-secrets ---
    print(f"🔍 Running detect-secrets scan...")
    rc, out, err = run_cmd(["detect-secrets","scan","--all-files","--json"], repo_path)
    print(f"   detect-secrets exit code: {rc}, output length: {len(out)}, errors: {len(err)}")
    if out.strip():
        try:
            data = json.loads(out)
            for file, secrets in data.get("results", {}).items():
                for s in secrets:
                    await add("detect-secrets", [{
                        "severity": "high",
                        "file": file,
                        "line": s.get("line_number"),
                        "rule_id": s.get("type"),
                        "message": f"Potential secret: {s.get('hashed_secret','***')}",
                    }])
        except Exception:
            pass

    # --- pip-audit ---
    req_file = None
    for cand in ["requirements.txt","backend/requirements.txt"]:
        if (repo_path / cand).exists():
            req_file = cand
            break
    if req_file:
        print(f"🔍 Running pip-audit dependency scan...")
        rc, out, err = run_cmd(["pip-audit","-r",req_file,"--format","json"], repo_path)
        print(f"   pip-audit exit code: {rc}, output length: {len(out)}, errors: {len(err)}")
        if out.strip():
            try:
                data = json.loads(out)
                for v in data.get("dependencies", []):
                    name = v.get("name"); version = v.get("version")
                    for adv in v.get("vulns", []):
                        await add("pip-audit", [{
                            "severity": adv.get("severity","MEDIUM"),
                            "file": req_file,
                            "line": None,
                            "rule_id": ",".join(adv.get("aliases",[])) or adv.get("id"),
                            "message": f"{name}=={version}: {adv.get('description','vulnerability')}",
                            "remediation": f"Upgrade {name} to {adv.get('fix_versions',["latest"])[0]}",
                            "autofixable": True
                        }])
            except Exception:
                pass

    # --- ruff (lint) ---
    print(f"🔍 Running ruff code quality analysis...")
    rc, out, err = run_cmd(["ruff","check","--output-format","json","."], repo_path)
    print(f"   ruff exit code: {rc}, output length: {len(out)}, errors: {len(err)}")
    if out.strip():
        try:
            data = json.loads(out)
            for item in data:
                await add("ruff", [{
                    "severity": "low",
                    "file": item.get("filename"),
                    "line": item.get("location",{}).get("row"),
                    "rule_id": item.get("code"),
                    "message": item.get("message"),
                    "autofixable": True
                }])
        except Exception:
            pass

    # --- mypy (types) ---
    rc, out, err = run_cmd(["mypy","--hide-error-codes","--no-error-summary","--pretty","--no-color-output","--no-error-summary","--explicit-package-bases","."], repo_path)
    # mypy doesn't do JSON by default; keep simple parse
    if out:
        for line in out.splitlines():
            if ":" in line and "error:" in line:
                try:
                    file, ln, _ = line.split(":", 2)
                    await add("mypy", [{
                        "severity": "medium",
                        "file": file.strip(),
                        "line": int(ln.strip()),
                        "rule_id": "mypy",
                        "message": line.strip(),
                    }])
                except Exception:
                    pass

    # --- radon (complexity) ---
    rc, out, err = run_cmd(["radon","cc","-j","."], repo_path)
    if out.strip():
        try:
            data = json.loads(out)
            for file, entries in data.items():
                for ent in entries:
                    if ent.get("rank") in ("D","E","F"):
                        await add("radon", [{
                            "severity": "medium" if ent.get("rank")=="D" else "high",
                            "file": file,
                            "line": ent.get("lineno"),
                            "rule_id": f"complexity-{ent.get('rank')}",
                            "message": f"Cyclomatic complexity {ent.get('complexity')} (rank {ent.get('rank')})",
                        }])
        except Exception:
            pass

    # --- npm audit ---
    if (repo_path / "package.json").exists():
        rc, out, err = run_cmd(["npm","audit","--json"], repo_path)
        if out.strip():
            try:
                data = json.loads(out)
                advisories = data.get("vulnerabilities", {})
                for name, v in advisories.items():
                    sev = v.get("severity","low")
                    await add("npm-audit", [{
                        "severity": sev,
                        "file": "package.json",
                        "line": None,
                        "rule_id": name,
                        "message": f"{name}: {v.get('title','vulnerability')}"
                    }])
            except Exception:
                pass

    # LLM enrichment (optional)
    context = {"repo": str(repo_path)}
    findings = enrich_findings_with_llm(findings, context)

    print(f"💾 Storing {len(findings)} findings in database")
    
    # Update progress before storing
    async with SessionLocal() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        job.current_stage, job.progress = "storing", 80
        await session.commit()
    
    # Store in DB
    async with SessionLocal() as session:
        for f in findings:
            session.add(Finding(**f))
        # Commit the findings first
        await session.commit()
        print(f"✅ Findings committed to database")
        
        # Update job status
        result = await session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        job.status, job.current_stage, job.progress = "completed", "done", 100
        await session.commit()
        print(f"🎉 Job {job_id} completed successfully!")
        
        # 📁 Repository kept in workspace for potential further work
        # It will be automatically cleaned up when starting work on a new repository
        print(f"📁 Repository {job_id} kept in workspace for potential further work")
        print(f"📁 It will be automatically cleaned up when starting work on a new repository")
