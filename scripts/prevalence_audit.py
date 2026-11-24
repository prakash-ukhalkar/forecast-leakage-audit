"""Prevalence audit of evaluation errors in public Python forecasting projects.

Subcommands
  collect   search GitHub (public API, unauthenticated) for forecasting projects in three popularity strata
  scan      shallow-clone a seeded random sample, run the static detectors, delete the clone
  summarize aggregate to repository level with Wilson intervals (aggregate results only)

Ethics: only public repositories are read, nothing is executed, clones are deleted after scanning, per-repository findings stay in a
local file (etc/prevalence_findings.jsonl) and only aggregate numbers and anonymised repository identifiers are reported.

Usage: python scripts/prevalence_audit.py collect | scan --n 160 | summarize
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sdg_leakage_audit.audit.static import analyse_file  # noqa: E402

OUT = ROOT / "results" / "prevalence"
FINDINGS = ROOT / "etc" / "prevalence_findings.jsonl"
QUERIES = ["time series forecasting", "stock price prediction", "energy consumption forecasting", "electricity load forecasting",
           "air quality prediction", "weather forecasting", "sales forecasting", "traffic forecasting", "lstm time series", "demand forecasting"]
STRATA = {"low": "stars:3..20", "mid": "stars:21..100", "high": "stars:101..3000"}
FAMILIES = ["RANDOM_SPLIT", "NONCHRONO_CV", "FIT_BEFORE_SPLIT", "LOOKAHEAD", "TEST_SELECTION"]
UA = {"User-Agent": "leakage-prevalence-audit", "Accept": "application/vnd.github+json"}


def api(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            wait = 30 * (attempt + 1)
            print("api retry", attempt, repr(exc)[:80], f"sleep {wait}s", flush=True)
            time.sleep(wait)
    return {"items": []}


def collect() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in QUERIES:
        for name, st in STRATA.items():
            qs = urllib.parse.quote(f"{q} language:python {st} fork:false archived:false")
            d = api(f"https://api.github.com/search/repositories?q={qs}&sort=stars&order=desc&per_page=100&page=1")
            for it in d.get("items", []):
                rows.append({"full_name": it["full_name"], "stars": it["stargazers_count"], "stratum": name, "size_kb": it["size"],
                             "created": it["created_at"][:10], "pushed": it["pushed_at"][:10], "query": q})
            print(q, name, len(d.get("items", [])), flush=True)
            time.sleep(7)  # unauthenticated search limit: 10 requests per minute
    df = pd.DataFrame(rows).drop_duplicates("full_name")
    df.to_csv(OUT / "candidates.csv", index=False)
    print("candidates", len(df), df.stratum.value_counts().to_dict())


def clone_scan(full_name: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="repo_"))
    res = {"full_name": full_name, "ok": False}
    try:
        env = {**__import__("os").environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
        p = subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--filter=blob:limit=2000000", "-q",
                            f"https://github.com/{full_name}.git", str(tmp / "r")], capture_output=True, text=True, timeout=120, env=env)
        if p.returncode != 0:
            res["error"] = p.stderr[:200]
            return res
        root = tmp / "r"
        res["is_package"] = any((root / f).exists() for f in ("setup.py", "pyproject.toml", "setup.cfg"))
        files = [f for f in list(root.rglob("*.py")) + list(root.rglob("*.ipynb"))
                 if not any(part in {"site-packages", "venv", ".venv", "node_modules", "tests", "test", ".git"} for part in f.parts) and f.stat().st_size < 2_000_000]
        res["n_files"] = len(files)
        fam_files = {k: 0 for k in FAMILIES}
        comp: dict[str, int] = {}
        evidence: dict[str, list[str]] = {k: [] for k in FAMILIES}
        n_qual = 0
        for f in files[:600]:
            try:
                rep = analyse_file(f.read_text(encoding="utf-8", errors="ignore"), f.suffix == ".ipynb")
            except Exception:  # noqa: BLE001
                continue
            if not rep.qualifies:
                continue
            n_qual += 1
            for k in FAMILIES:
                if k in rep.flags:
                    fam_files[k] += 1
                    if len(evidence[k]) < 3:
                        evidence[k].append(f"{f.relative_to(root)} :: {rep.flags[k][0]}")
            for k, v in rep.compliant.items():
                comp[k] = comp.get(k, 0) + v
        res.update({"n_qualifying_files": n_qual, "fam_files": fam_files, "compliant": comp, "evidence": evidence, "ok": True})
    except Exception as exc:  # noqa: BLE001
        res["error"] = repr(exc)[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def scan(n: int, seed: int) -> None:
    cand = pd.read_csv(OUT / "candidates.csv")
    rng = random.Random(seed)
    names = sorted(cand.full_name)
    rng.shuffle(names)
    done = {}
    if FINDINGS.exists():
        for ln in FINDINGS.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln)
            done[d["full_name"]] = d
    todo = [x for x in names if x not in done][: max(0, n - len(done))]
    print(f"{len(done)} already scanned, scanning {len(todo)}", flush=True)
    FINDINGS.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(4) as ex, FINDINGS.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(clone_scan, todo), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(i, flush=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(seed: int) -> None:
    cand = pd.read_csv(OUT / "candidates.csv").set_index("full_name")
    rows = []
    for ln in FINDINGS.read_text(encoding="utf-8").splitlines():
        d = json.loads(ln)
        if not d.get("ok"):
            continue
        rows.append({"full_name": d["full_name"], "is_package": d["is_package"], "n_files": d["n_files"], "n_qual": d["n_qualifying_files"],
                     **{k: int(d["fam_files"][k] > 0) for k in FAMILIES}, **{f"files_{k}": d["fam_files"][k] for k in FAMILIES},
                     "ts_cv": int(d["compliant"].get("time_series_cv", 0) + d["compliant"].get("time_series_split", 0) > 0),
                     "shuffle_false": int(d["compliant"].get("split_shuffle_false", 0) > 0),
                     "slice_split": int(d["compliant"].get("slice_split", 0) > 0)})
    df = pd.DataFrame(rows)
    df["stratum"] = df.full_name.map(cand.stratum)
    df["stars"] = df.full_name.map(cand.stars)
    n_scanned = len(df)
    Q = df[(df.n_qual > 0) & (~df.is_package)].copy()
    rng = random.Random(seed)
    ids = list(range(1, len(Q) + 1))
    rng.shuffle(ids)
    Q["repo_id"] = [f"R{i:03d}" for i in ids]
    Q["any_error"] = Q[FAMILIES].max(axis=1)
    Q["n_error_families"] = Q[FAMILIES].sum(axis=1)
    Q.drop(columns=["full_name", "stars", "n_files", "n_qual"] + [f"files_{k}" for k in FAMILIES]).sort_values("repo_id").to_csv(OUT / "repo_level_anonymised.csv", index=False)
    pd.DataFrame({"full_name": sorted(df.full_name)}).to_csv(OUT / "scanned_repositories.csv", index=False)
    summ = []
    for k in [*FAMILIES, "any_error"]:
        lo, hi = wilson(int(Q[k].sum()), len(Q))
        summ.append({"family": k, "repos": int(Q[k].sum()), "of": len(Q), "share_pct": 100 * Q[k].mean(), "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    pd.DataFrame(summ).to_csv(OUT / "prevalence_summary.csv", index=False)
    byst = Q.groupby("stratum")[[*FAMILIES, "any_error"]].mean().mul(100).round(1)
    byst["n"] = Q.groupby("stratum").size()
    byst.to_csv(OUT / "prevalence_by_stratum.csv")
    comp = {"time_series_cv": Q.ts_cv.mean() * 100, "shuffle_false_split": Q.shuffle_false.mean() * 100, "slice_split": Q.slice_split.mean() * 100}
    pd.Series(comp).round(1).to_csv(OUT / "compliant_practices.csv", header=["share_pct"])
    print(f"scanned {n_scanned}; non-package with a forecasting-evaluation file: {len(Q)}")
    print(pd.DataFrame(summ).round(1).to_string(index=False))
    print(byst.to_string())
    print(comp)
    print("mean error families per repo:", Q.n_error_families.mean().round(2), "share with >=2:", (Q.n_error_families >= 2).mean().round(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "scan", "summarize"])
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    {"collect": collect, "scan": lambda: scan(a.n, a.seed), "summarize": lambda: summarize(a.seed)}[a.cmd]()


if __name__ == "__main__":
    main()
