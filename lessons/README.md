# Track lessons

Lessons that belong to a subject track rather than to one industry programme. Each is a
notebook you do, with the same exercises, checks and hints as every other lesson here; the
worked solution and full rubric come with enrolment on Synapsa. Where a track has more lessons
than exist so far, the rest are planned, not specified in this repository yet.

| Track | Lesson | What you build | Open |
|---|---|---|---|
| T00 · orientation | [`T00-L01-the-8gb-track`](T00-L01-the-8gb-track/) | The profiler and the tier gate that decide whether any lesson in this repository may ship. Most programmes start here. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T00-L01-the-8gb-track/lesson.ipynb) |
| T00 · orientation | [`T00-L02-model-registry`](T00-L02-model-registry/) | A role-based model registry, a retirement detector run against a dated snapshot of a provider's model pages, and a golden test that survives a model swap. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T00-L02-model-registry/lesson.ipynb) |
| T00 · orientation | [`T00-L03-one-client-three-backends`](T00-L03-one-client-three-backends/) | The normalising half of a provider-agnostic LLM client, which reads tool calls, reasoning and JSON out of whatever shape a backend returns, and a compatibility suite that shows what three stand-in backends keep, rename, drop or reject. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T00-L03-one-client-three-backends/lesson.ipynb) |
| T03 · language models | [`T03-L01-bpe-from-scratch`](T03-L01-bpe-from-scratch/) | Byte-level byte pair encoding from scratch, and the token tax it imposes on five writing systems. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T03-L01-bpe-from-scratch/lesson.ipynb) |
| T03 · language models | [`T03-L02-bpe-merge-loop-in-cpp`](T03-L02-bpe-merge-loop-in-cpp/) | The BPE merge loop in C++, and its speed-up over the Python trainer on one identical corpus. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T03-L02-bpe-merge-loop-in-cpp/lesson.ipynb) |
| T03 · language models | [`T03-L03-nanolm-cpu-slice`](T03-L03-nanolm-cpu-slice/) | A character-level language model trained on a CPU with numpy alone, its backward pass derived by hand and checked against finite differences. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T03-L03-nanolm-cpu-slice/lesson.ipynb) |
| T04 · post-training | [`T04-L01-grpo-from-scratch`](T04-L01-grpo-from-scratch/) | GRPO from scratch in numpy on a tiny model, and its traps measured: zero-variance groups, length and standard-deviation normalisation, group size and a reward hack. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T04-L01-grpo-from-scratch/lesson.ipynb) |
| T05 · inference systems | [`T05-L01-serving-on-a-cpu`](T05-L01-serving-on-a-cpu/) | A small transformer served on a CPU the way an inference engine does: a KV cache and static batching, with the operation, memory and latency accounting that shows what each buys. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T05-L01-serving-on-a-cpu/lesson.ipynb) |
| T05 · inference systems | [`T05-L02-paged-kv-and-continuous-batching`](T05-L02-paged-kv-and-continuous-batching/) | A paged KV cache and a continuous-batching scheduler for a small language model, proof that they change no token, and what they buy against a contiguous cache with static batching. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T05-L02-paged-kv-and-continuous-batching/lesson.ipynb) |
| T06 · retrieval and RAG | [`T06-L01-retrieval-from-scratch`](T06-L01-retrieval-from-scratch/) | BM25, a dense retriever with no external model, and hybrid fusion, scored by an evaluation harness whose intervals say when one retriever is really better. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T06-L01-retrieval-from-scratch/lesson.ipynb) |
| T06 · retrieval and RAG | [`T06-L02-chunking-and-context-budget`](T06-L02-chunking-and-context-budget/) | Chunking and context assembly under a token budget, measured by whether the evidence an answer needs actually reaches the model. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T06-L02-chunking-and-context-budget/lesson.ipynb) |
| T06 · retrieval and RAG | [`T06-L03-reranking-measured`](T06-L03-reranking-measured/) | A learned reranker, pointwise and pairwise, whose gain is stated with its interval, what reranking depth costs, and the trap of tuning on test queries. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T06-L03-reranking-measured/lesson.ipynb) |
| T06 · retrieval and RAG | [`T06-L04-agent-memory`](T06-L04-agent-memory/) | Memory for a long-running assistant: what to store, how recency, importance and relevance combine at retrieval, and why an updated fact must beat the stale one. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T06-L04-agent-memory/lesson.ipynb) |
| T07 · agents and protocols | [`T07-L01-agent-loop-from-scratch`](T07-L01-agent-loop-from-scratch/) | An agent harness with validated tools, budgets, retries and traces, and a measurement of how per-step reliability compounds over long tasks. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T07-L01-agent-loop-from-scratch/lesson.ipynb) |
| T07 · agents and protocols | [`T07-L02-mcp-from-the-wire-up`](T07-L02-mcp-from-the-wire-up/) | A Model Context Protocol server and client built from the specification, checked by a conformance harness built from the same specification. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T07-L02-mcp-from-the-wire-up/lesson.ipynb) |
| T07 · agents and protocols | [`T07-L03-evaluating-agents`](T07-L03-evaluating-agents/) | An agent evaluation whose numbers carry intervals: pass^k against pass@k, trajectory metrics, a failure taxonomy from traces, and a paired comparison of two configurations. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T07-L03-evaluating-agents/lesson.ipynb) |
| T07 · agents and protocols | [`T07-L04-agent-to-agent-a2a`](T07-L04-agent-to-agent-a2a/) | An Agent2Agent (A2A) Agent Card validator, task-lifecycle server and streaming client built from the specification, checked by a conformance harness built from the same specification. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T07-L04-agent-to-agent-a2a/lesson.ipynb) |
| T07 · agents and protocols | [`T07-L05-constrained-decoding`](T07-L05-constrained-decoding/) | A JSON-schema constrained decoder: a token mask over a BPE vocabulary, and a measurement of what the mask does to the output distribution. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T07-L05-constrained-decoding/lesson.ipynb) |
| T08 · AI engineering | [`T08-L01-agent-observability`](T08-L01-agent-observability/) | The agent loop instrumented with OpenTelemetry GenAI spans, and a cost-per-task report rebuilt from the spans alone. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T08-L01-agent-observability/lesson.ipynb) |
| T10 · evaluation and governance | [`T10-L01-ai-act-conformity-pack`](T10-L01-ai-act-conformity-pack/) | An AI system registry turned into a machine-checkable EU AI Act conformity evidence pack. The EU AI Act programme starts here. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T10-L01-ai-act-conformity-pack/lesson.ipynb) |
| T10 · evaluation and governance | [`T10-L02-evaluating-across-languages`](T10-L02-evaluating-across-languages/) | The instruments that evaluate a system across six generated languages: token fertility, cost per correct answer, paired bootstrap intervals, aggregation traps, a translation-artefact detector and a power curve. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T10-L02-evaluating-across-languages/lesson.ipynb) |
| T10 · evaluation and governance | [`T10-L03-a-mini-arena`](T10-L03-a-mini-arena/) | A mini arena's ranking engine: Bradley-Terry ratings from pairwise votes, a position-bias fit, bootstrap intervals, and the number of votes before neighbouring systems stop swapping places. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T10-L03-a-mini-arena/lesson.ipynb) |
| T12 · AI for science | [`T12-L01-find-gw150914`](T12-L01-find-gw150914/) | The first gravitational wave ever detected, found in the raw LIGO data on a laptop: noise spectra, whitening, a chirp template and matched filtering, from scratch. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T12-L01-find-gw150914/lesson.ipynb) |
| T12 · AI for science | [`T12-L02-rainfall-runoff-honestly`](T12-L02-rainfall-runoff-honestly/) | A four-parameter rainfall-runoff model, a calibrator and a lagged-rain baseline, with the hold-out, split and flow-band traps measured on two real Australian catchments. | [Colab](https://colab.research.google.com/github/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials/blob/master/lessons/T12-L02-rainfall-runoff-honestly/lesson.ipynb) |

Each lesson's `meta.yaml` lists its prerequisites. T00-L01 and T10-L01 need none; beyond them:

- T00-L02 builds on T00-L01.
- T00-L03 builds on T00-L01.
- T03-L01 builds on T00-L01.
- T03-L02 builds on T03-L01.
- T03-L03 builds on T00-L01.
- T04-L01 builds on T00-L01.
- T05-L01 builds on T03-L03.
- T05-L02 builds on T03-L03 and T05-L01.
- T06-L01 builds on T00-L01.
- T06-L02 builds on T06-L01.
- T06-L03 builds on T06-L01 and T06-L02.
- T06-L04 builds on T06-L01 and T07-L01.
- T07-L01 builds on T00-L01.
- T07-L02 builds on T07-L01.
- T07-L03 builds on T07-L01.
- T07-L04 builds on T07-L01 and T07-L02.
- T07-L05 builds on T03-L01 and T07-L01.
- T08-L01 builds on T07-L01 and T07-L03.
- T10-L02 builds on T03-L01.
- T10-L03 builds on T00-L01.
- T12-L01 builds on T00-L01.
- T12-L02 builds on T00-L01.

<!-- NOTICE:BEGIN generated by tools/status.py; do not edit -->

---

**Synapsa Commons** · © 2026 RealAI · licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). You may use this material to learn and to teach, and copy, fork, share and adapt it. You must credit "Synapsa Commons by RealAI" with a link to <https://github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials>, say what you changed, and share adaptations under the same licence. You may not use it, or anything adapted from it, for commercial advantage or payment; for a commercial licence, contact [RealAI](https://www.realai.eu/contact). The full summary is in [NOTICE](../NOTICE); the licence's legal code governs.

<!-- NOTICE:END -->
