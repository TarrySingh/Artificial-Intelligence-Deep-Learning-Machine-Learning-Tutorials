#!/usr/bin/env python3
"""Are the "Before you start" links alive? Maintainer tool; needs the network.

    python tools/check_resources.py [lesson-dir ...]     # default: every lesson

Reads each lesson's `before_you_start.resources` in meta.yaml and fetches every url (follows
redirects). 2xx is fine. A 403 or 429 from a publisher that turns scripts away is reported as
"blocked", which needs a person to open the link once; anything else is a failure. Whether a
resource really is free, and really covers what its `why` says, is for the reviewer: a live link
proves only that the page exists.
"""
import concurrent.futures as cf, sys, urllib.request, urllib.error
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (compatible; SynapsaCommonsLinkCheck/1.0; +https://synapsa.realai.eu)"


def lessons(args):
    if args:
        return [Path(a).resolve() for a in args]
    out = []
    for pat in ("lessons/*/", "flagships/*/lessons/*/", "programmes/*/lessons/*/"):
        out += [p for p in ROOT.glob(pat) if (p / "meta.yaml").exists()]
    return sorted(out)


def probe(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.status, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, url
    except Exception as e:                      # DNS, TLS, timeout
        return 0, f"{type(e).__name__}: {e}"[:80]


def main():
    jobs = []
    for d in lessons(sys.argv[1:]):
        meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
        for r in (meta.get("before_you_start") or {}).get("resources") or []:
            jobs.append((d.name, r["url"]))
    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda j: probe(j[1]), jobs))
    bad = blocked = 0
    for (name, url), (code, note) in zip(jobs, results):
        if 200 <= code < 300:
            continue
        if code in (403, 429):
            blocked += 1
            print(f"  blocked {code}  {name}  {url}")
        else:
            bad += 1
            print(f"  FAIL {code}  {name}  {url}  {note if code == 0 else ''}")
    print(f"\n  {len(jobs)} links: {len(jobs) - bad - blocked} live, {blocked} blocked (open once by hand), {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
