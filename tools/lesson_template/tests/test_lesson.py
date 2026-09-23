"""Autograder rubric. Hidden cases live here, not in the notebook.

Each entry: (callable, points, hint shown on failure).
Partial credit is mandatory — never collapse the rubric into one all-or-nothing test.
"""
import importlib.util, os
from pathlib import Path

# CI grades solutions/lesson_solution.py; a student grades lesson.py. Same rubric, one file.
_src = Path(__file__).resolve().parents[1] / os.environ.get("COMMONS_LESSON_SRC", "lesson.py")
_spec = importlib.util.spec_from_file_location("lesson", _src)
lesson = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson)


def test_basic():
    assert lesson.exercise_one(2) == 4


def test_edge_zero():
    assert lesson.exercise_one(0) == 0


def test_negative():
    assert lesson.exercise_one(-3) == -6


RUBRIC = [
    (test_basic, 4, "start with the worked example in the docstring"),
    (test_edge_zero, 3, "what does your code do with 0?"),
    (test_negative, 3, "negative inputs are not a special case here"),
]
