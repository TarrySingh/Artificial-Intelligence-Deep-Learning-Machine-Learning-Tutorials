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
