#!/usr/bin/env python3
"""Execution gate (gates 9 and 10). Stdlib only.

    python tools/execute.py lessons/<lesson-id> [--write-back]

Runs `solutions/lesson_solution.py` (the reference implementation, which must pass the
lesson's own asserts) in a fresh interpreter, measures wall time and peak RSS of the child,
and compares them against `budget_seconds` and the tier in meta.yaml. With --write-back it
records the MEASURED numbers into meta.yaml so no human types a performance figure.
"""
import argparse, re, resource, subprocess, sys, time
from pathlib import Path

TIERS = {"phone": 2048, "browser": 2048, "cpu8": 8192, "free-gpu": 16384,
         "gpu24": 24576, "gpu80": 81920, "multi-gpu": 163840, "api": 8192}


def read_meta(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"^([a-z_]+):\s*(.+?)\s*$", line)
            if m:
                # strip inline comments: `tier: cpu8   # phone | browser | ...`
                val = re.sub(r"\s+#.*$", "", m.group(2)).strip().strip('"\'')
                out[m.group(1)] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lesson")
    ap.add_argument("--write-back", action="store_true")
    a = ap.parse_args()
    lesson = Path(a.lesson).resolve()
    meta_path = lesson / "meta.yaml"
    meta = read_meta(meta_path)
    tier = meta.get("tier", "cpu8")
    budget = float(meta.get("budget_seconds", 600))
    # A compiled lesson must build FRESH. make sees a binary newer than its source and calls
    # it up to date, even when the library path baked into it is dead — which is exactly what
    # happened when this tree moved and every C binary still pointed at the old venv's
    # libmujoco. Deleting the artefacts is cheaper than trusting a timestamp.
    if meta.get("language", "python") in {"c", "cpp"}:
        for pat in ("lesson_bin", "solutions/lesson_bin", "*.o", "solutions/*.o"):
            for stale in lesson.glob(pat):
                stale.unlink(missing_ok=True)
        for d in (lesson / "build", lesson / "solutions" / "build"):
            if d.is_dir():
                import shutil as _sh
                _sh.rmtree(d, ignore_errors=True)

    target = lesson / "solutions" / "lesson_solution.py"
    if not target.exists():
        target = lesson / "lesson.py"
    if not target.exists():
        print(f"FAIL {lesson.name}: nothing to execute")
        return 1

    # Measure the child's OWN peak, inside a fresh wrapper process, so the number is a
    # property of this lesson and not of whatever else the shell has spawned. RUSAGE_SELF
    # covers the lesson; RUSAGE_CHILDREN covers anything it shells out to (make, clang, a
    # compiled binary). A cumulative delta across runs is meaningless — that was the old bug.
    wrapper = (
        "import resource, runpy, sys\n"
        "try:\n"
        "    runpy.run_path(sys.argv[1], run_name='__main__')\n"
        "finally:\n"
        "    _s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        "    _c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss\n"
        "    sys.stderr.write('__COMMONS_PEAK__%d\\n' % max(_s, _c))\n"
    )
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", wrapper, target.name], cwd=target.parent,
                          capture_output=True, text=True, timeout=budget * 3)
    wall = time.monotonic() - t0
    m = re.search(r"__COMMONS_PEAK__(\d+)", proc.stderr)
    raw = int(m.group(1)) if m else 0
    # ru_maxrss: bytes on macOS, kibibytes on Linux
    peak_mib = raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024

    ok = proc.returncode == 0
    within_time = wall <= budget
    if tier not in TIERS:
        print(f"FAIL {lesson.name}: unknown tier {tier!r}; declare one of {sorted(TIERS)}")
        return 1
    within_mem = peak_mib <= TIERS[tier]
    status = "PASS" if (ok and within_time and within_mem) else "FAIL"
    print(f"{status} {lesson.name}: exit={proc.returncode} wall={wall:.1f}s "
          f"(budget {budget:.0f}s) peak={peak_mib:.0f}MiB (tier {tier}) ")
    if not ok:
        print("--- stderr tail ---")
        tail = [l for l in proc.stderr.strip().splitlines() if "__COMMONS_PEAK__" not in l]
        print("\n".join(tail[-15:]))
    if a.write_back and ok:
        text = meta_path.read_text() if meta_path.exists() else ""
        for key, val in (("measured_seconds", f"{wall:.1f}"), ("measured_peak_mib", f"{peak_mib:.0f}")):
            if re.search(rf"^{key}:", text, re.M):
                text = re.sub(rf"^{key}:.*$", f"{key}: {val}", text, flags=re.M)
            else:
                text += f"\n{key}: {val}"
        meta_path.write_text(text.rstrip() + "\n")
        print(f"  wrote measured numbers back to {meta_path.name}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
