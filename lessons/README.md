# Track lessons

Lessons that belong to a subject track rather than to one industry programme. Each is a
notebook you do, with the same exercises, checks, hints, rubric and worked solution as every
other lesson here. Where a track has more lessons than exist so far, the rest are planned, not
specified in this repository yet.

| Track | Lesson | What you build | Open |
|---|---|---|---|
| T00 · compute | [`T00-L01-the-8gb-track`](T00-L01-the-8gb-track/) | The profiler and the tier gate that decide whether any lesson in this repository may ship. Most programmes start here. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T00-L01-the-8gb-track/lesson.ipynb) |
| T03 · language models | [`T03-L01-bpe-from-scratch`](T03-L01-bpe-from-scratch/) | Byte-level byte pair encoding from scratch, and the token tax it imposes on five writing systems. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T03-L01-bpe-from-scratch/lesson.ipynb) |
| T03 · language models | [`T03-L02-bpe-merge-loop-in-cpp`](T03-L02-bpe-merge-loop-in-cpp/) | The BPE merge loop in C++, and its speed-up over the Python trainer on one identical corpus. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T03-L02-bpe-merge-loop-in-cpp/lesson.ipynb) |
| T03 · language models | [`T03-L03-nanolm-cpu-slice`](T03-L03-nanolm-cpu-slice/) | A character-level language model trained on a CPU with numpy alone, its backward pass derived by hand and checked against finite differences. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T03-L03-nanolm-cpu-slice/lesson.ipynb) |
| T10 · regulation | [`T10-L01-ai-act-conformity-pack`](T10-L01-ai-act-conformity-pack/) | An AI system registry turned into a machine-checkable EU AI Act conformity evidence pack. The EU AI Act programme starts here. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T10-L01-ai-act-conformity-pack/lesson.ipynb) |

Each lesson's `meta.yaml` lists its prerequisites: T03-L01 and T03-L03 build on T00-L01, and
T03-L02 builds on T03-L01.
