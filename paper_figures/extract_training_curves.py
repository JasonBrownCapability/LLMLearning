"""Extract per-step training curves from the archived wandb logs.

The original training runs used ``save_strategy="no"`` and predate the in-flight
``_save_trainer_state`` instrumentation, so ``trainer_state.json`` is only on disk
for condition B. However, every run's wandb ``output.log`` contains the full
per-step metric dicts that TRL prints at each ``logging_steps`` interval. This
script parses those logs straight out of ``results/wandb_logs.tar.gz`` (in-memory,
no extraction) and writes a small, committable JSON of the curves so that
``build_figures.py`` can render Figure 3 without touching the tarball.

Run from the repo root:
    python paper_figures/extract_training_curves.py

Produces:
    paper_figures/training_curves.json

Selection (Llama-3.1 8B):
    B : GRPO run, peak lr ~1e-5, full 2000 steps   -> reward climbs to ~0.75
    C : GRPO run, peak lr ~1e-6, full 2000 steps   -> reward plateaus ~0.5
    D : SFT  run, full 2000 steps                  -> CE loss + token accuracy

Condition G (distillation) used a custom AdamW loop that did not log to this
wandb project, so its KL-loss curve is not recoverable here; it remains
prose-described in Appendix E.
"""

from __future__ import annotations

import ast
import json
import re
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARBALL = REPO_ROOT / "results" / "wandb_logs.tar.gz"
OUT = Path(__file__).resolve().parent / "training_curves.json"

RUN_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")
# A logged metric line is a Python dict repr containing 'loss' or 'reward'.
METRIC_KEYS = (
    "loss", "reward", "kl", "grad_norm", "learning_rate",
    "entropy", "mean_token_accuracy",
)


def _to_float(v):
    """Best-effort float conversion of a wandb-logged string value."""
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_log(text: str) -> list[dict]:
    """Pull the per-step metric dicts out of one run's output.log."""
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not (s.startswith("{") and s.endswith("}")):
            continue
        if "'loss'" not in s and "'reward'" not in s:
            continue
        try:
            d = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            continue
        row = {k: _to_float(d[k]) for k in METRIC_KEYS if k in d}
        if row:
            rows.append(row)
    return rows


def _logging_steps(config_text: str | None, default: int = 10) -> int:
    if not config_text:
        return default
    m = re.search(r"logging_steps:\s*(?:\n\s*value:\s*)?([0-9]+)", config_text)
    return int(m.group(1)) if m else default


def _series(rows, key):
    return [r.get(key) for r in rows]


def _finite(xs):
    return [x for x in xs if isinstance(x, (int, float))]


def load_runs() -> dict[str, dict]:
    """Read every run's parsed metric rows + logging_steps from the tarball."""
    runs: dict[str, dict] = {}
    logs: dict[str, str] = {}
    cfgs: dict[str, str] = {}
    with tarfile.open(TARBALL, "r:gz") as tar:
        for m in tar.getmembers():
            mm = RUN_RE.search(m.name)
            if not mm or not m.isfile():
                continue
            rid = mm.group(1)
            if m.name.endswith("output.log"):
                logs[rid] = tar.extractfile(m).read().decode("utf-8", "ignore")
            elif m.name.endswith("config.yaml"):
                cfgs[rid] = tar.extractfile(m).read().decode("utf-8", "ignore")
    for rid, text in logs.items():
        rows = _parse_log(text)
        if not rows:
            continue
        keys = set().union(*(set(r) for r in rows))
        kind = "grpo" if "reward" in keys else ("sft" if "mean_token_accuracy" in keys else "other")
        lrs = _finite(_series(rows, "learning_rate"))
        runs[rid] = {
            "run_id": rid,
            "kind": kind,
            "n": len(rows),
            "logging_steps": _logging_steps(cfgs.get(rid)),
            "peak_lr": max(lrs) if lrs else None,
            "rows": rows,
        }
    return runs


def _curve(run: dict) -> dict:
    """Materialise a run's selected series with an explicit step axis."""
    rows = run["rows"]
    ls = run["logging_steps"]
    step = [(i + 1) * ls for i in range(len(rows))]
    out = {"run_id": run["run_id"], "kind": run["kind"], "step": step}
    keys = ("reward", "kl", "grad_norm", "loss", "mean_token_accuracy")
    for k in keys:
        s = _series(rows, k)
        if any(isinstance(v, (int, float)) for v in s):
            out[k] = s
    return out


def select(runs: dict[str, dict]) -> dict:
    """Pick the representative B, C, D 8B runs by log content, deterministically.

    B : longest GRPO run with peak lr >= 5e-6  (standard 1e-5 schedule)
    C : longest GRPO run with peak lr <  5e-6  (10x-reduced 1e-6 schedule)
    D : longest *single* SFT run (one 2000-step training run, not a
        multi-run log where several trainings share one process/output.log)
    Ties broken by run_id for reproducibility.
    """
    # A single training run logs at most max_steps/logging_steps points; with
    # max_steps=2000 and logging_steps=10 that is ~200. Logs with many more
    # points concatenate several runs and must not be used for a single curve.
    SINGLE_RUN_MAX = 210

    grpo = sorted((r for r in runs.values() if r["kind"] == "grpo"),
                  key=lambda r: (-r["n"], r["run_id"]))
    sft = sorted((r for r in runs.values()
                  if r["kind"] == "sft" and r["n"] <= SINGLE_RUN_MAX),
                 key=lambda r: (-r["n"], r["run_id"]))

    def first(seq, pred):
        return next((r for r in seq if pred(r)), None)

    b = first(grpo, lambda r: (r["peak_lr"] or 0) >= 5e-6)
    c = first(grpo, lambda r: (r["peak_lr"] or 1) < 5e-6)
    d = sft[0] if sft else None

    picked = {}
    if b:
        picked["B"] = _curve(b)
    if c:
        picked["C"] = _curve(c)
    if d:
        picked["D"] = _curve(d)
    return picked


def main():
    if not TARBALL.exists():
        raise SystemExit(f"missing {TARBALL}")
    runs = load_runs()
    picked = select(runs)

    payload = {
        "_about": (
            "Per-step training curves for Llama-3.1 8B conditions B/C/D, parsed "
            "from results/wandb_logs.tar.gz output.log files. Step axis is "
            "reconstructed as (line_index+1) * logging_steps. Condition G "
            "(distillation) used a custom loop not logged to this wandb project "
            "and is not included."
        ),
        "conditions": picked,
    }
    OUT.write_text(json.dumps(payload, indent=2))

    # Auditable report of what was selected.
    print(f"parsed {len(runs)} runs with metric logs; selected:")
    for cond in ("B", "C", "D"):
        c = picked.get(cond)
        if not c:
            print(f"  {cond}: (none found)")
            continue
        steps = c["step"][-1]
        if c["kind"] == "grpo":
            rw = _finite(c["reward"])
            gn = _finite(c["grad_norm"])
            tail = sum(rw[-10:]) / len(rw[-10:])
            print(f"  {cond}: run {c['run_id']} grpo, {steps} steps, "
                  f"reward tail~{tail:.3f}, grad-norm med~{sorted(gn)[len(gn)//2]:.2f}")
        else:
            ls = _finite(c["loss"])
            ta = _finite(c.get("mean_token_accuracy", []))
            ta_s = f", token-acc {ta[0]:.3f}->{ta[-1]:.3f}" if ta else ""
            print(f"  {cond}: run {c['run_id']} sft, {steps} steps, "
                  f"loss {ls[0]:.3f}->{ls[-1]:.3f}{ta_s}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
