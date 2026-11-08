#!/usr/bin/env python3
"""Compare the C++ core against the golden fixture (CONTROL_SPEC §8).

Builds nothing: it expects ``str_core_trace`` to exist already.

    cmake -S . -B build && cmake --build build
    python3 tests/golden/compare.py --trace build/str_core_trace

Exit code 0 means every compared column agrees within tolerance.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "step_response_case.csv"


def parse_fixture(path: Path):
    meta: dict[str, str] = {}
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append({k: float(v) for k, v in row.items()})
    with path.open(encoding="utf-8") as fh:
        for ln in fh:
            if not ln.startswith("#"):
                break
            body = ln[1:].strip()
            if body.startswith("config:"):
                meta["config"] = body[len("config:"):].strip()
            elif body.startswith("theta0="):
                meta["theta0"] = body[len("theta0="):].strip()
            elif ":" in body and "=" not in body:
                key, _, val = body.partition(":")
                meta.setdefault(key.strip(), val.strip())
    return meta, rows


def trace_args(meta: dict[str, str]) -> list[str]:
    cfg = meta.get("config", "")
    vals: dict[str, str] = {}
    for tok in cfg.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            vals[k] = v
    theta0 = meta.get("theta0", "")
    args = ["--order", str(int(float(vals.get("order", "2")))),
            "--h", vals["h"], "--lambda", vals["lambda"], "--pmax", vals["pmax"],
            "--zeta", vals["zeta"], "--wn", vals["wn"], "--mu", vals["mu"],
            "--umax", vals["umax"], "--percent", vals["percent"],
            "--theta0", theta0]
    return args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="build/str_core_trace")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    if not FIXTURE.exists():
        print(f"missing fixture {FIXTURE}; run generate_golden.py first", file=sys.stderr)
        return 2
    if not Path(args.trace).exists():
        print(f"missing trace binary {args.trace}; build with cmake first", file=sys.stderr)
        return 2

    meta, rows = parse_fixture(FIXTURE)
    stdin_text = "".join(f"{r['y_des']:.17g} {r['y']:.17g}\n" for r in rows)
    cmd = [str(args.trace), *trace_args(meta)]
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)
    if proc.returncode != 0:
        print("C++ trace failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr.strip()[:2000], file=sys.stderr)
        return 1

    lines = [ln for ln in proc.stdout.splitlines() if ln and not ln.startswith("#")]
    if not lines:
        print("trace produced no rows", file=sys.stderr)
        return 1
    header = lines[0].split(",")
    out = []
    for ln in lines[1:]:
        vals = [float(v) for v in ln.split(",")]
        out.append(dict(zip(header, vals)))

    if len(out) != len(rows):
        print(f"row count mismatch: fixture {len(rows)} vs trace {len(out)}", file=sys.stderr)
        return 1

    # Column correspondence: fixture name -> trace name
    pairs = [
        ("u_design", "u_design"),
        ("u_cmd", "u_cmd"),
        ("theta0", "theta0"), ("theta1", "theta1"),
        ("theta2", "theta2"), ("theta3", "theta3"),
    ]
    worst: list[tuple[str, float, float]] = []
    ok = True
    for fx, cx in pairs:
        if cx not in header:
            print(f"trace missing column {cx!r}; has {header}", file=sys.stderr)
            return 1
        scale = max(abs(r[fx]) for r in rows) or 1.0
        worst_abs, worst_tick = 0.0, -1
        for i, (fr, cr) in enumerate(zip(rows, out)):
            d = abs(fr[fx] - cr[cx])
            if d > worst_abs:
                worst_abs, worst_tick = d, i
        rel = worst_abs / scale
        worst.append((fx, worst_abs, rel))
        if rel > args.tol:
            ok = False
            print(f"  FAIL {fx}: max|Δ|={worst_abs:.3e} (rel {rel:.3e}) at tick {worst_tick}")

    print(f"{'column':<12}{'max|Δ|':>14}{'rel':>14}")
    for name, a, r in worst:
        flag = "" if r <= args.tol else "  <-- FAIL"
        print(f"{name:<12}{a:>14.3e}{r:>14.3e}{flag}")
    print(f"\nn={len(rows)} ticks, tol={args.tol:g}, u range "
          f"[{min(r['u_cmd'] for r in out):.4f}, {max(r['u_cmd'] for r in out):.4f}]")
    print("PASS: Python and C++ cores agree" if ok else "FAIL: cores diverge")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
