# The AI Atlas — wave 1 (pre-alpha, unpublished)

Local build only. Nothing here has been pushed to any GitHub repository, and the repo
strategy and history purge are still open decisions.

- `QUALITY.md` — the 12 gates every lesson must pass, and the language policy.
- `tools/` — the autograder (`grade.py`), the execution gate (`execute.py`), lesson template.
- `flagships/` — flagship subtrees, each with its own README and lessons.
- `lessons/` — track lessons.
- `programmes/` — industry-vertical programmes.

Run a lesson's grader:   `python tools/grade.py flagships/<id>/lessons/<lesson>`
Run the execution gate:  `python tools/execute.py flagships/<id>/lessons/<lesson> --write-back`
Verify the whole repo:   `python tools/verify_all.py`

## What exists today

**31 lessons, all passing all four gates.** Every `MODULES.md` marks a module BUILT only when a
directory exists and the gates pass on it; everything else says `specified`, and means it.

| Area | Built | Specified, not built |
|---|---|---|
| `flagships/humanoid-lab` — humanoid locomotion in MuJoCo | 8 + capstone | — |
| `lessons/` — track openers (T00, T03, T10) | 5 | the rest of every track |
| `programmes/ai-act-conformity` | 4 of 9 | 5 |
| `programmes/document-intelligence` | 5 of 11 | 6 |
| `programmes/model-risk` | 5 of 10 | 5 |
| `programmes/predictive-maintenance` | 4 of 9 | 5 |

Five of the 31 are compiled C or C++ exercises rather than notebooks. C# is deferred until a
.NET runner exists, per `QUALITY.md`.
